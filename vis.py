import torch
import numpy as np
from skew_normal_class import delta_given_Omega_and_alpha_with_normalization, affine_skew_normal, retrieve_skew_normal_parameters, sample_skew_normal, mode_of_skew_normal, skew_normal_pdf
import matplotlib.pyplot as plt
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def plot_2d_skew_normal(xi, Omega, alpha, x_min, x_max, y_min, y_max, steps=100, ax=None, num=20, curve=True, ratio=0.8, cmap='viridis', transparency=1.):
    xs = torch.linspace(x_min, x_max, steps)
    ys = torch.linspace(y_min, y_max, steps)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    pdf = skew_normal_pdf(torch.stack([grid_x, grid_y], dim=-1), xi, Omega, alpha)
    if ax is None:
        if curve:
            # plt.contour(grid_x.numpy(), grid_y.numpy(), pdf.numpy(), num, cmap=cmap)
            plt.contour(grid_x.numpy(), grid_y.numpy(), pdf.numpy(), num, levels=np.linspace(pdf.max() * ratio, pdf.max(), num), cmap=cmap)
        else:
            plt.contourf(grid_x.numpy(), grid_y.numpy(), pdf.numpy(), num, cmap=cmap, levels=np.linspace(pdf.max() * ratio, pdf.max(), num))
    else:
        if curve:
            ax.contour(grid_x.numpy(), grid_y.numpy(), pdf.numpy(), num, levels=np.linspace(pdf.max() * ratio, pdf.max(), num), alpha=transparency, cmap=cmap)
            # ax.contour(grid_x.numpy(), grid_y.numpy(), pdf.numpy(), num, levels=np.linspace(pdf.max() * ratio, pdf.max(), 4))
        else:
            ax.contourf(grid_x.numpy(), grid_y.numpy(), pdf.numpy(), num, levels=np.linspace(pdf.max() * ratio, pdf.max(), num), alpha=transparency, cmap=cmap)


def draw_2d_skew_normal_w_axis():
    # Generate a random 2D skew-normal distribution
    logits = torch.randn(7)
    xi, Omega, alpha = retrieve_skew_normal_parameters(logits)
    # scale the skewness
    scale_factor = 10
    alpha = alpha * scale_factor
    # draw samples from the skew-normal distribution
    x = sample_skew_normal(xi, Omega, alpha, (100,))
    x_min = x[..., 0].min().item()
    x_max= x[..., 0].max().item()
    y_min = x[..., 1].min().item()
    y_max = x[..., 1].max().item()
    # plot the pdf of the skew-normal distribution
    plot_2d_skew_normal(xi, Omega, alpha, x_min, x_max, y_min, y_max, ratio=0.5)
    # plot the principal axes, delta, and the mode of the skew-normal distribution
    eigenvalues, eigenvectors = torch.linalg.eigh(Omega)
    eigenvalues = eigenvalues.sqrt()
    x_axis, y_axis = eigenvectors
    mean = mode_of_skew_normal(xi, Omega, alpha, False)
    xi_aff, Omega_aff, alpha_aff = affine_skew_normal(mean, Omega, alpha, eigenvectors, torch.zeros_like(alpha))
    plt.scatter(mean[0], mean[1], c='k')
    pos_x_axis = torch.stack([mean, mean + x_axis * eigenvalues[0]], dim=1)
    pos_y_axis = torch.stack([mean, mean + y_axis * eigenvalues[1]], dim=1)
    neg_x_axis = torch.stack([mean, mean - x_axis * eigenvalues[0]], dim=1)
    neg_y_axis = torch.stack([mean, mean - y_axis * eigenvalues[1]], dim=1)
    alpha_aff = torch.linalg.inv(eigenvectors) @ alpha_aff
    pos_alpha_aff = torch.stack([mean, mean + alpha_aff], dim=1)
    neg_alpha_aff = torch.stack([mean, mean - alpha_aff], dim=1)
    delta = delta_given_Omega_and_alpha_with_normalization(Omega, alpha)
    pos_delta = torch.stack([mean, mean + delta], dim=1)
    neg_delta = torch.stack([mean, mean - delta], dim=1)
    for axis in [pos_x_axis, pos_y_axis, neg_x_axis, neg_y_axis]:
        plt.arrow(axis[0, 0], axis[1, 0], axis[0, 1] - axis[0, 0], axis[1, 1] - axis[1, 0], width=0.005, head_width=0.05, facecolor='b', edgecolor='b')
    for axis in [pos_delta]:
        plt.arrow(axis[0, 0], axis[1, 0], axis[0, 1] - axis[0, 0], axis[1, 1] - axis[1, 0], width=0.005, head_width=0.05, facecolor='orange', edgecolor='orange')
    plt.axis('equal')
    plt.savefig('.cache/2d_skew_normal_w_axis_mode.png')


if __name__ == '__main__':
    draw_2d_skew_normal_w_axis()
