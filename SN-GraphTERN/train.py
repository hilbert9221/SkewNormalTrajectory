import os
import pickle
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
from graphtern import *
from utils import *
from torch.utils.data import DataLoader
import time
from normalize import translate_and_rotate, inv_translate_and_rotate
from log import create_logger
from time import strftime, gmtime

# Reproducibility
seed = int(time.time())
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# torch.manual_seed(0)
# random.seed(0)
# np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Argument parsing
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_epgcn', type=int, default=1, help='Number of EPGCN layers for endpoint prediction')
parser.add_argument('--n_epcnn', type=int, default=6, help='Number of EPCNN layers for endpoint prediction')
parser.add_argument('--n_trgcn', type=int, default=1, help='Number of TRGCN layers for trajectory refinement')
parser.add_argument('--n_trcnn', type=int, default=3, help='Number of TRCNN layers for trajectory refinement')
parser.add_argument('--n_ways', type=int, default=3, help='Number of control points for endpoint prediction')
parser.add_argument('--n_smpl', type=int, default=20, help='Number of samples for refine')
parser.add_argument('--kernel_size', type=int, default=3)

# Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='zara1', help='Dataset name(eth,hotel,univ,zara1,zara2)')

# Training specifc parameters
parser.add_argument('--batch_size', type=int, default=128, help='Mini batch size')
parser.add_argument('--num_epochs', type=int, default=512, help='Number of epochs')
parser.add_argument('--clip_grad', type=float, default=None, help='Gradient clipping')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=128, help='Number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=False, help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', help='Personal tag for the model')
parser.add_argument('--dist', default='normal', help='Personal tag for the model', choices=['normal', 'skew'])
parser.add_argument('--date', default='',)
parser.add_argument('--dont_normalize', action='store_true', help='Do not normalize the data')

args = parser.parse_args()
if args.tag == 'tag':
    args.tag = f'{args.dist}_{args.dataset}'

# Data preparation
# Batch size set to 1 because vertices vary by humans in each scene sequence.
# Use mini batch working like batch.
if len(args.date) == 0:
    args.date = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
dataset_path = './datasets/' + args.dataset + '/'
checkpoint_dir = f'./checkpoint/{args.date}_{args.dist}_graph-tern_{args.dataset}/'
# checkpoint_dir = './checkpoint/' + args.tag + '/'
my_log = create_logger(__name__, silent=False, to_disk=True, log_file=f'{checkpoint_dir}/log.txt')
my_log.info(str(args))

train_dataset = TrajectoryDataset(dataset_path + 'train/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = TrajectoryDataset(dataset_path + 'val/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = graph_tern(n_epgcn=args.n_epgcn, n_epcnn=args.n_epcnn, n_trgcn=args.n_trgcn, n_trcnn=args.n_trcnn,
                   seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len, n_ways=args.n_ways, n_smpl=args.n_smpl, dist=args.dist)
model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
if args.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.8)

# Train logging
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir + 'args.pkl', 'wb') as f:
    pickle.dump(args, f)

metrics = {'train_loss': [], 'val_loss': [],}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10,}
# metrics = {'train_loss': [], 'val_loss': [], 'val_ade': [], 'val_fde': []}
# constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10, 'min_val_ade': 1e10, 'min_val_fde': 1e10, 'min_val_epoch_loss': -1}


def train(epoch):
    global metrics, model
    model.train()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(train_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        # sum gradients till idx reach to batch_size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]

        # Data augmentation
        aug = True
        if aug:
            S_obs, S_trgt = data_sampler(S_obs, S_trgt, batch=1)
        if not args.dont_normalize:
            S_obs, S_trgt, origin, angle, scale = translate_and_rotate(S_obs, S_trgt)
            # Run Graph-TERN model
            V_init, V_pred, V_refi, valid_mask = model(S_obs, S_trgt)

            # Loss calculation
            r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways, dist=args.dist)
            S_obs, S_trgt, V_refi = inv_translate_and_rotate(S_obs, S_trgt, V_refi, origin, angle, scale)
        else:
            V_init, V_pred, V_refi, valid_mask = model(S_obs, S_trgt)
            r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways, dist=args.dist)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask)
        loss = r_loss + m_loss

        if torch.isnan(loss):
            pass
        else:
            loss.backward()
            loss_batch += loss.item()

        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Train Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()
    metrics['train_loss'].append(loss_batch / loader_len)


def valid(epoch):
    global metrics, constant_metrics, model
    model.eval()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(val_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(val_loader):
        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]
        if not args.dont_normalize:
            S_obs, S_trgt, origin, angle, scale = translate_and_rotate(S_obs, S_trgt)
            # Run Graph-TERN model
            V_init, V_pred, V_refi, valid_mask = model(S_obs)

            # Loss calculation
            r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways, dist=args.dist)
            S_obs, S_trgt, V_refi = inv_translate_and_rotate(S_obs, S_trgt, V_refi, origin, angle, scale)
        else:
            V_init, V_pred, V_refi, valid_mask = model(S_obs)
            r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways, dist=args.dist)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask, training=False)
        loss = r_loss + m_loss

        loss_batch += loss.item()
        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()
    metrics['val_loss'].append(loss_batch / loader_len)

    # Save model
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + args.dataset + '_best.pth')


def valid_ade_fde(epoch):
    global metrics, constant_metrics, model
    model.eval()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(val_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    model.n_smpl = 20
    ade_refi_all = []
    fde_refi_all = []

    for batch_idx, batch in enumerate(val_loader):
        S_obs, S_trgt = [tensor.cuda() for tensor in batch[-2:]]

        # Run Graph-TERN model
        V_init, V_pred, V_refi, valid_mask = model(S_obs)

        # Loss calculation
        r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways, dist=args.dist)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask, training=False)
        loss = r_loss + m_loss

        loss_batch += loss.item()
        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            r_loss_batch = 0.
            m_loss_batch = 0.

        # Calculate ADEs and FDEs for each refined trajectory
        V_trgt_abs = S_trgt[:, 0].squeeze(dim=0)
        temp = (V_refi - V_trgt_abs).norm(p=2, dim=-1)
        ADEs = temp.mean(dim=1).min(dim=0)[0]
        FDEs = temp[:, -1, :].min(dim=0)[0]
        ade_refi_all.extend(ADEs.tolist())
        fde_refi_all.extend(FDEs.tolist())

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()
    metrics['val_loss'].append(loss_batch / loader_len)

    ade_refi = sum(ade_refi_all) / len(ade_refi_all)
    fde_refi = sum(fde_refi_all) / len(fde_refi_all)
    metrics['val_ade'].append(ade_refi)
    metrics['val_fde'].append(fde_refi)

    # Save model
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch_loss'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + args.dataset + '_best_loss.pth')
    if (metrics['val_ade'][-1] + metrics['val_fde'][-1]) < (constant_metrics['min_val_ade'] + constant_metrics['min_val_fde']):
        constant_metrics['min_val_ade'] = metrics['val_ade'][-1]
        constant_metrics['min_val_fde'] = metrics['val_fde'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + args.dataset + '_best.pth')

def main():
    for epoch in range(args.num_epochs):
        train(epoch)
        valid(epoch)

        if args.use_lrschd:
            scheduler.step()

        my_log.info(" ")
        my_log.info("Dataset: {0}, Epoch: {1}".format(args.tag, epoch))
        my_log.info("Train_loss: {0}, Val_los: {1}".format(metrics['train_loss'][-1], metrics['val_loss'][-1]))
        my_log.info("Min_val_epoch: {0}, Min_val_loss: {1}".format(constant_metrics['min_val_epoch'], constant_metrics['min_val_loss']))

        with open(checkpoint_dir + 'metrics.pkl', 'wb') as f:
            pickle.dump(metrics, f)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)


if __name__ == "__main__":
    main()
