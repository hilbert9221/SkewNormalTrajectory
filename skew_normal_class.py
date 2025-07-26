'''
Reorganize skew_normal.py into different classes

BUG: No validate_args now.
'''
from torch.distributions import Distribution
import torch
import math
import numpy as np
import scipy
from functools import partial
from general import pool_map
import torch.nn.functional as F
from torch import nn


class MultiNormal(Distribution):
    '''
    My implementation of Multivariate Normal Distribution instead of using torch.distributions.MultivariateNormal to better support batch operations
    '''
    def __init__(self, mu=None, Sigma=None, logits=None):
        if mu is not None:
            self.mu = mu
            self.Sigma = Sigma
        elif logits is not None:
            self.retrieve_parameters(logits)
        else:
            raise ValueError('Either mu and Sigma or logits must be provided.')
        super().__init__(self.mu.shape[:-1], self.mu.shape[-1:], False)
        
    def retrieve_parameters(self, logits, eps=1e-20):
        '''
        Args:
            logits: [..., (d + 3) * d / 2]
        '''
        # (- b + sqrt(b^2 - 4ac)) / 2a
        d = int((- 3 + math.sqrt(9 + 8 * logits.size(-1))) // 2)
        mu = logits[..., :d]
        Sigma = construct_nd_covariance_matrix(logits[..., d:], epsilon=eps)
        self.mu, self.Sigma = mu, Sigma
    
    def nll(self, y, add_constant=True, unify=False):
        mu, Sigma = self.mu, self.Sigma
        eps = 1e-20
        delta = y - mu
        if int(y.shape[-1]) == 2:
            det = det_2x2_matrix(Sigma)
            inv_Sigma = inv_2x2_matrix(Sigma)
        else:
            det = torch.linalg.det(Sigma)
            inv_Sigma = torch.linalg.inv(Sigma)
        quadratic_form = 1 / 2 * compute_quadratic_form(delta, inv_Sigma, delta)
        
        if unify:
            if add_constant:
                k = int(y.shape[-1])
                nll = - ((- quadratic_form).exp() / ((2 * math.pi) ** (k / 2) * det.clamp(min=1e-20).sqrt())).clamp(min=eps).log()
            else:
                nll = - ((- quadratic_form).exp() / det.clamp(min=1e-20).sqrt()).clamp(min=eps).log()
        else:
            nll = quadratic_form + det.clamp(min=eps).log() / 2
            if add_constant:
                k = int(y.shape[-1])
                const = k / 2 * math.log(2 * math.pi)
                nll = nll + const
        return nll
    
    def log_prob(self, y, add_constant=True, unify=False):
        return - self.nll(y, add_constant, unify)
    
    def sample(self, shape=torch.Size()):
        return self.rsample(shape)
    
    def rsample(self, shape=torch.Size()):
        return sample_normal(self.mu, self.Sigma, shape)
    
    def pdf(self, y):
        mu, Sigma = self.mu, self.Sigma
        return normal_pdf(y, mu, Sigma)

    def entropy(self,):
        return normal_entropy(self.Sigma)


def normal_entropy(Sigma):
    d = Sigma.size(-1)
    constant = (1 + math.log(2 * math.pi)) * d / 2
    log_det = torch.det(Sigma).clamp(min=1e-20).log()
    return constant + log_det / 2


class DiagNormal(Distribution, nn.Module):
    def __init__(self, mu=None, sigma=None, logits=None):
        if mu is not None:
            self.mu = mu
            # sigma means sigma^2 indeed
            self.sigma = sigma
        elif logits is not None:
            self.retrieve_parameters(logits)
        else:
            raise ValueError('Either mu and Sigma or logits must be provided.')
        super().__init__(self.mu.shape[:-1], self.mu.shape[-1:], False)

    def retrieve_parameters(self, logits, eps=1e-20):
        '''
        Args:
            logits: [..., 2 * d]
        '''
        d = logits.size(-1) // 2
        mu = logits[..., :d]
        sigma = logits[..., d:].exp().clamp(min=eps)
        self.mu, self.sigma = mu, sigma

    def weighted_sum(self, weights):
        '''
        Args:
            mu: [..., N, dim]
            weights: [..., K, N]
        '''
        mu, sigma = self.mu, self.sigma
        mu = weights @ mu
        sigma = (weights ** 2) @ sigma
        return mu, sigma

    def sample(self, shape=torch.Size()):
        return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        mu, sigma = self.mu, self.sigma
        dim = int(mu.shape[-1])
        noise = torch.randn(*(shape + mu.shape[:-1]), dim).to(mu.device)
        mu = left_expand(mu, len(shape))
        sigma_sqrt = left_expand(sigma.clamp(min=1e-20).sqrt(), len(shape))
        return mu + noise * sigma_sqrt
    
    def pdf(self, y):
        mu, Sigma = self.mu, torch.diag_embed(self.sigma)
        return normal_pdf(y, mu, Sigma)

    def nll(self,):
        ...

    def log_prob(self, y):
        return - self.nll(y)


class DiagSkewNormal(Distribution):
    def __init__(self, xi=None, omega=None, alpha=None, logits=None):
        if xi is not None:
            self.xi = xi
            self.omega = omega
            self.alpha = alpha
        elif logits is not None:
            self.retrieve_parameters(logits)
        else:
            raise ValueError('Either mu and Sigma or logits must be provided.')
        super().__init__(self.xi.shape[:-1], self.xi.shape[-1:], False)

    @property
    def mean(self,):
        return mean_of_skew_normal(self.xi, torch.diag_embed(self.omega), self.alpha)
    
    @property
    def cov(self,):
        return covariance_of_skew_normal(torch.diag_embed(self.omega), self.alpha)
    
    @property
    def variance(self,):
        return self.cov

    def retrieve_parameters(self, logits, eps=1e-20):
        '''
        Args:
            logits: [..., 2 * d + d]
        '''
        d = logits.size(-1) // 3
        xi = logits[..., :d]
        omega = F.softplus(logits[..., d:2 * d]).clamp(min=eps)
        # omega = logits[..., d:2 * d].exp().clamp(min=eps)
        alpha = logits[..., 2 * d:]
        self.xi, self.omega, self.alpha = xi, omega, alpha

    def compute_delta(self,):
        delta = (self.alpha / (1 + self.alpha ** 2).sqrt()).sum(-1)
        return delta.mean(), delta.numel()

    def sample_wo_alpha(self, shape=torch.Size()):
        mu, sigma = self.mu, self.omega
        dim = int(mu.shape[-1])
        noise = torch.randn(*(shape + mu.shape[:-1]), dim).to(mu.device)
        mu = left_expand(mu, len(shape))
        sigma_sqrt = left_expand(sigma.clamp(min=1e-20).sqrt(), len(shape))
        return mu + noise * sigma_sqrt

    def sample(self, shape=torch.Size()):
        return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        xi, omega, alpha = self.xi, self.omega, self.alpha
        dim = int(xi.shape[-1])
        scale = 1 / torch.sqrt(1 + (alpha ** 2).sum(-1, keepdim=True))
        delta = (scale * alpha).unsqueeze(-1)
        omega_bar = torch.diag_embed(torch.ones_like(omega))
        omega_aug = torch.cat([delta, omega_bar], dim=-1)
        delta_aug = torch.cat([torch.ones(*xi.shape[:-1], 1, 1, device=xi.device), delta.transpose(-1, -2).contiguous()], dim=-1)
        omega_aug = torch.cat([delta_aug, omega_aug], dim=-2)
        noise = torch.randn(*(shape + xi.shape[:-1]), dim + 1).to(xi.device)
        # omega_sqrt = torch.linalg.cholesky(omega_aug)
        omega_sqrt = cholesky_with_exception(omega_aug)
        noise = (omega_sqrt @ noise.unsqueeze(-1)).squeeze(-1)
        x0 = noise[..., 0]
        x = noise[..., 1:]
        z = torch.zeros_like(x)
        indicator = x0 > 0
        z[indicator] = x[indicator]
        z[~indicator] = - x[~indicator]
        xi = left_expand(xi, len(shape))
        omega_sqrt = left_expand(omega_sqrt, len(shape))
        scale = omega.clamp(min=1e-20).sqrt()
        scale = left_expand(scale, len(shape))
        sample = xi + z * scale
        # t1 = time.time()
        # print(t1 - t0)
        return sample

    def nll(self, y, add_constant=True, approx_cdf=False, unify=False):
        return nll_skew_normal(y, self.xi, torch.diag_embed(self.omega), self.alpha, add_constant, approx_cdf, unify)

    def log_prob(self, x):
        return - self.nll(x)
    
    def entropy(self,):
        return skew_normal_entropy(torch.diag_embed(self.omega), self.alpha)


class SkewNormal(Distribution):
    def __init__(self, xi=None, Omega=None, alpha=None, logits=None):
        if xi is not None:
            self.xi = xi
            self.Omega = Omega
            self.alpha = alpha
        elif logits is not None:
            self.retrieve_parameters(logits)
        else:
            raise ValueError('Either mu and Sigma or logits must be provided.')
        super().__init__(self.xi.shape[:-1], self.xi.shape[-1:], False)
        
    def retrieve_parameters(self, logits, eps=1e-20):
        '''
        Args:
            logits: [..., (d + 5) * d / 2]
        '''
        # (- b + sqrt(b^2 - 4ac)) / 2a
        d = int((- 5 + math.sqrt(25 + 8 * logits.size(-1))) // 2)
        xi = logits[..., :d]
        alpha = logits[..., d:2 * d]
        if d == 2:
            sigma = logits[..., 2 * d:2 * d + 2].exp()
            rho = logits[..., 2 * d + 2].tanh()
            Omega = construct_covariance_matrix(sigma, rho)
        else:
            Omega = construct_nd_covariance_matrix(logits[..., 2 * d:], epsilon=eps)
        self.xi, self.Omega, self.alpha = xi, Omega, alpha
    
    def log_prob(self, y, add_constant=True, approx_cdf=False, unify=False):
        return - self.nll(y, add_constant, approx_cdf, unify)

    def nll(self, y, add_constant=True, approx_cdf=False, unify=False):
        return nll_skew_normal(y, self.xi, self.Omega, self.alpha, add_constant, approx_cdf, unify)
    
    def sample(self, shape=torch.Size(), option='reparameterization'):
        if option == 'reparameterization':
            return self.rsample(shape)
        elif option == 'additive':
            return self.asample(shape)
        else:
            raise ValueError('Invalid option.')
    
    def rsample(self, shape=torch.Size()):
        xi, Omega, alpha = self.xi, self.Omega, self.alpha
        # create augmented covariance matrix
        eps = 1e-20
        dim = int(xi.shape[-1])
        alpha = alpha.unsqueeze(-1)
        omega = batch_diag(Omega).clamp(min=eps).sqrt()
        # eta = alpha / omega.unsqueeze(-1)
        inv_omega_diag_mat = torch.diag_embed(1. / omega)
        Omega_bar = inv_omega_diag_mat @ Omega @ inv_omega_diag_mat
        scale = 1 / torch.sqrt(1 + alpha.transpose(-1, -2).contiguous() @ Omega_bar @ alpha)
        delta = scale * Omega_bar @ alpha
        # delta = delta_given_omega_and_alpha(Omega_bar, eta.transpose(-1, -2).contiguous())
        omega_aug = torch.cat([delta, Omega_bar], dim=-1)
        delta_aug = torch.cat([torch.ones(*xi.shape[:-1], 1, 1, device=xi.device), delta.transpose(-1, -2).contiguous()], dim=-1)
        omega_aug = torch.cat([delta_aug, omega_aug], dim=-2)
        noise = torch.randn(*(shape + xi.shape[:-1]), dim + 1).to(xi.device)
        omega_sqrt = cholesky_with_exception(omega_aug)
        noise = (omega_sqrt @ noise.unsqueeze(-1)).squeeze(-1)
        x0 = noise[..., 0]
        x = noise[..., 1:]
        z = torch.zeros_like(x)
        indicator = x0 > 0
        z[indicator] = x[indicator]
        z[~indicator] = - x[~indicator]
        xi = left_expand(xi, len(shape))
        omega = left_expand(omega, len(shape))
        sample = xi + z * omega
        return sample

    def asample(self, shape=torch.Size()):
        mu, Omega, alpha = self.xi, self.Omega, self.alpha
        # create augmented covariance matrix
        dim = int(xi.shape[-1])
        alpha = alpha.unsqueeze(-1)
        omega = batch_diag(Omega).clamp(min=1e-20).sqrt()
        # eta = alpha / omega.unsqueeze(-1)
        inv_omega_diag_mat = torch.diag_embed(1. / omega)
        Omega_bar = inv_omega_diag_mat @ Omega @ inv_omega_diag_mat
        scale = 1 / torch.sqrt(1 + alpha.transpose(-1, -2).contiguous() @ Omega_bar @ alpha)
        delta = scale * Omega_bar @ alpha
        D_delta = (1 - delta.squeeze(-1) ** 2).sqrt()
        D_delta_mat = torch.diag_embed(D_delta)
        inv_D_delta = torch.diag_embed(1. / D_delta)
        lambda_delta = inv_D_delta @ delta
        Psi_bar = inv_D_delta @ Omega_bar @ inv_D_delta - lambda_delta @ lambda_delta.transpose(-1, -2).contiguous()
        omega_aug = torch.cat([torch.zeros_like(delta), Psi_bar], dim=-1)
        delta_aug = torch.cat([torch.ones(*xi.shape[:-1], 1, 1, device=xi.device), torch.zeros_like(delta).transpose(-1, -2).contiguous()], dim=-1)
        omega_aug = torch.cat([delta_aug, omega_aug], dim=-2)
        noise = torch.randn(*(shape + xi.shape[:-1]), dim + 1).to(xi.device)
        omega_sqrt = cholesky_with_exception(omega_aug)
        noise = (omega_sqrt @ noise.unsqueeze(-1)).squeeze(-1)
        x0 = noise[..., [0]]
        x = noise[..., 1:]
        D_delta_mat = left_expand(D_delta_mat, len(shape))
        delta = left_expand(delta, len(shape))
        z = (D_delta_mat @ x.unsqueeze(-1)).squeeze(-1) + delta.squeeze(-1) * x0.abs()
        xi = left_expand(xi, len(shape))
        omega = left_expand(omega, len(shape))
        sample = xi + z * omega
        return sample
    
    def pdf(self, x):
        xi, Omega, alpha = self.xi, self.Omega, self.alpha
        pdf = normal_pdf(x, xi, Omega)
        Omega_sqrt = batch_diag(Omega).clamp(min=1e-20).sqrt()
        inner_product = ((x - xi) * (alpha / Omega_sqrt)).sum(-1)
        cdf = cdf_normal(inner_product)
        return 2 * pdf * cdf

def skew_normal_pdf(x, xi, Omega, alpha):
    '''
    Args:
        x: [..., dim]
        mu: [..., dim]
        Sigma: [..., dim, dim]
        alpha: [..., dim]
    '''
    pdf = normal_pdf(x, xi, Omega)
    Omega_sqrt = batch_diag(Omega).sqrt()
    inner_product = ((x - xi) * (alpha / Omega_sqrt)).sum(-1)
    cdf = cdf_normal(inner_product)
    return 2 * pdf * cdf

def skew_normal_entropy(Omega, alpha):
    entropy_normal = normal_entropy(Omega)
    Omega_bar = get_Omega_bar(Omega)
    sigma = compute_quadratic_form(alpha, Omega_bar, alpha).clamp(min=1e-20).sqrt()
    num = 1000
    shape = (num,)
    x0 = right_expand(torch.randn(*shape).to(sigma.device), len(sigma.shape)) * torch.ones(*(shape+sigma.shape)).to(sigma.device)
    noise = right_expand(torch.randn(*shape).to(sigma.device), len(sigma.shape)) * torch.ones(*(shape+sigma.shape)).to(sigma.device)
    samples = sample_univariate_skew_normal_given_noise(torch.zeros_like(sigma), torch.ones_like(sigma), sigma, x0, noise)
    expectation = math.log(2) + cdf_normal(sigma[None,] * samples).log().mean(0)
    return entropy_normal - expectation


def sample_univariate_skew_normal_given_noise(xi, omega, alpha, x0, noise, shape=torch.Size()):
    '''
    BUG: NOT IMPLEMENTED YET
    Args:
        shape: tuple
        xi: [..., 1]
        omega: [..., 1]
        alpha: [..., 1]

    Return:
        sample: [shape + xi.shape, ..., 1]
    '''
    # x0 = torch.randn(*(shape + xi.shape)).to(xi.device)
    # noise = torch.randn(*(shape + xi.shape)).to(xi.device)
    omega_sqrt = omega.clamp(min=1e-20).sqrt()
    # eta = alpha / omega_sqrt
    alpha = left_expand(alpha, len(shape))
    x1 = (alpha * x0 - noise) / (1 + alpha ** 2).sqrt()
    z = torch.zeros_like(x0)
    indicator = x1 > 0
    z[indicator] = x0[indicator]
    z[~indicator] = - x0[~indicator]
    xi = left_expand(xi, len(shape))
    omega_sqrt = left_expand(omega_sqrt, len(shape))
    sample = xi + z * omega_sqrt
    return sample


def nll_skew_normal(y, xi, Omega, alpha, add_constant=True, approx_cdf=False, unify=True):
        eps = 1e-20
        delta = y - xi
        omega = batch_diag(Omega).clamp(min=eps).sqrt()
        eta = alpha / omega.clamp(min=eps)
        Omega_bar = get_Omega_bar(Omega)
        if int(y.shape[-1]) == 2:
            det_Omega_bar = det_2x2_matrix(Omega_bar)
            inv_Omega = inv_2x2_matrix(Omega)
        else:
            det_Omega_bar = torch.linalg.det(Omega_bar)
            inv_Omega = torch.linalg.inv(Omega)
        quadratic_form = 1 / 2 * compute_quadratic_form(delta, inv_Omega, delta)
        # log \Phi
        inner_product = (eta * delta).sum(-1)
        if approx_cdf:
            cdf = approx_cdf_normal(inner_product)
        else:
            cdf = cdf_normal(inner_product)
        if unify:
            if add_constant:
                k = int(y.shape[-1])
                pdf = (2 * (- quadratic_form).exp() / ((2 * math.pi) ** (k / 2) * det_Omega_bar.clamp(min=eps).sqrt()) / (omega.prod(-1) * cdf).clamp(min=eps))
                nll = - pdf.clamp(min=eps).log()
            else:
                nll = - ((- quadratic_form).exp() / det_Omega_bar.clamp(min=eps).sqrt() / (omega.prod(-1) * cdf).clamp(min=eps)).clamp(min=eps).log()
        else:
            nll = quadratic_form + det_Omega_bar.clamp(min=eps).log() / 2 + omega.prod(-1).clamp(min=eps).log() - cdf.clamp(min=eps).log()
            if add_constant:
                k = int(y.shape[-1])
                const_pdf = k / 2 * math.log(2 * math.pi)
                const_cdf = - math.log(2)
                const = const_pdf + const_cdf
                nll = nll + const
        return nll


def r_nu_1st(kappa):
    '''
    Args:
        kappa: [...]
    '''
    return torch.special.i1e(kappa) / torch.special.i0e(kappa)

def construct_covariance_matrix(sigma, rho, epsilon=1e-20):
    '''
    Args:
        sigma: [..., 2]
        rho: [...]
        epsilon: float, augment \Sigma as \Sigma + \epsilon I, 1e-20 originall, 1e-8 now
    Return:
        : [..., 2, 2]
    '''
    sigma_up = sigma[..., 0]
    sigma_down = sigma[..., 1]
    correlation = sigma_up * sigma_down * rho
    matrix = torch.stack([
        torch.stack([sigma_up ** 2 + epsilon, correlation], dim=-1),
        torch.stack([correlation, sigma_down ** 2 + epsilon], dim=-1),
    ], dim=-2)
    return matrix


def construct_nd_covariance_matrix(logits, epsilon=1e-20):
    '''
    D for sqrt(var), D(D-1)/2 for correlation coefficients

    Args:
        logits: [..., D(D+1)/2]
    '''
    k = math.floor(math.sqrt(int(logits.shape[-1]) * 2))
    # [..., D, D], diagonal matrix
    diag = F.softplus(logits[..., :k]).clamp(min=epsilon)
    # [..., D(D-1)/2]
    cov = logits[..., k:]
    # [..., D, D]
    L = torch.diag_embed(diag)
    # ones = torch.diag_embed(torch.ones_like(diag))# * epsilon
    # assign correlation coefficients to mat
    # multiply sigma and rho to get the valid covariance matrix
    start = 0
    for i in range(1, k):
        end = start + i
        L[..., i, :i] = cov[..., start:end]
        start = end
    mat = L @ L.transpose(-1, -2).contiguous()
    # avoid singularity
    # mat = mat + ones
    return mat


def left_expand(x, n):
    for _ in range(n):
        x = x.unsqueeze(0)
    return x


def right_expand(x, n):
    for _ in range(n):
        x = x.unsqueeze(-1)
    return x


def cholesky_with_exception(mat):
    '''
    Args:
        mat: [..., dim, dim]
    '''
    orthogonal, info = torch.linalg.cholesky_ex(mat)
    select = info != 0
    if select.any():
        eps = 1e-4
        dim = mat.size(-1)
        diagonal = eps * torch.eye(dim).to(mat.device)
        nonpositive = mat[select]
        diagonal = left_expand(diagonal, len(mat.shape) - 2)
        nonpositive = nonpositive + diagonal
        intermediate = torch.zeros_like(orthogonal)
        intermediate[~select] = orthogonal[~select]
        intermediate[select] = torch.linalg.cholesky(nonpositive)
        orthogonal = intermediate
    return orthogonal


'''
Instrumental functions
'''
def batch_diag(x):
    '''
    Args:
        x: [..., dim, dim]
    '''
    n = int(x.shape[-1])
    diag = torch.stack([
        x[..., i, i] for i in range(n)
    ], dim=-1)
    return diag


def det_2x2_matrix(x):
    '''
    NOTE: Pass test

    Args:
        x: [..., 2, 2]
    '''
    return x[..., 0, 0] * x[..., 1, 1] - x[..., 0, 1] * x[..., 1, 0]


def inv_2x2_matrix(x):
    # NOTE: Pass test
    det = det_2x2_matrix(x)
    adjoint = torch.stack([
        torch.stack([x[..., 1, 1], - x[..., 0, 1]], dim=-1),
        torch.stack([- x[..., 1, 0], x[..., 0, 0]], dim=-1),
    ], dim=-2)
    inverse = adjoint / right_expand(det, 2)
    return inverse


def compute_quadratic_form(x, mat, y):
    '''
    Args:
        x: [..., dim]
        mat: [..., dim, dim]
        y: [..., dim]
    '''
    return (x.unsqueeze(-2) @ mat @ y.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def batch_trace(x):
    '''
    NOTE: Pass test

    Args:
        x: [..., dim, dim]

    Return:
        trace: [...]
    '''
    eye = torch.eye(int(x.shape[-1])).to(x.device)
    eye = left_expand(eye, len(x.shape[:-2]))
    trace = (x * eye).sum((-1, -2))
    return trace


def get_Omega_bar(Omega):
    Omega_sqrt = batch_diag(Omega).clamp(min=1e-20).sqrt()
    inv_Omega_sqrt = 1. / Omega_sqrt
    inv_Omega_sqrt = torch.diag_embed(inv_Omega_sqrt)
    Omega_bar = inv_Omega_sqrt @ Omega @ inv_Omega_sqrt
    return Omega_bar


def fast_cholesky_factorization(y, omega, delta):
    '''
    Args:
        y: [..., dim + 1]
        omega, delta: [..., dim]
    '''
    lower_triangle = (omega - delta ** 2).clamp(min=1e-20).sqrt()
    ret = torch.cat([
        y[..., [0]], lower_triangle * y[..., 1:] + y[..., [0]] * delta
    ], dim=-1)
    return ret


def delta_given_Omega_and_alpha_with_normalization(Omega, alpha):
    '''
    delta is invariant to scale transformation.

    Args:
        Omega: correlation matrix
        alpha: normalized alpha
    '''
    Omega_bar = get_Omega_bar(Omega)
    denominator = (1 + compute_quadratic_form(alpha, Omega_bar, alpha)).sqrt()
    delta = (Omega_bar @ alpha.unsqueeze(-1)).squeeze(-1) / denominator.unsqueeze(-1)
    return delta


def affine_skew_normal(xi, Omega, alpha, affine, constant):
    '''
    Args:
        xi: [..., dim]
        Omega: [..., dim, dim]
        alpha: [..., dim]
        affine: [..., dim_out, dim]
        constant: [..., dim_out]
    '''
    xi_new = constant + (affine @ xi.unsqueeze(-1)).squeeze(-1)
    Omega_new = affine @ Omega @ affine.transpose(-1, -2).contiguous()
    delta = delta_given_Omega_and_alpha_with_normalization(Omega, alpha)
    omega = batch_diag(Omega).sqrt()
    omega_new = batch_diag(Omega_new).sqrt()
    affine_omega_delta = (affine @ (omega * delta).unsqueeze(-1)).squeeze(-1)
    inv_Omega_new = torch.linalg.inv(Omega_new)
    denominator = (1 - compute_quadratic_form(affine_omega_delta, inv_Omega_new, affine_omega_delta)).sqrt()
    alpha_new = (torch.diag_embed(omega_new) @ inv_Omega_new @ affine_omega_delta.unsqueeze(-1)).squeeze(-1) / denominator.unsqueeze(-1)
    return xi_new, Omega_new, alpha_new


def retrieve_skew_normal_parameters(logits, eps=1e-20, normalize=False):
    xi = logits[..., :2]
    if normalize:
        sigma = logits[..., 2:4]
        sigma = sigma - sigma.max(-1, keepdim=True)[0]
        sigma = sigma.exp()
    else:
        sigma = F.softplus(logits[..., 2:4])
        # sigma = logits[..., 2:4].exp()
    rho = logits[..., 4].tanh().clamp(min=-1 + eps, max=1 - eps)
    alpha = logits[..., 5:]
    if normalize:
        alpha = alpha.tanh()
    omega = construct_covariance_matrix(sigma, rho, eps)
    # omega = construct_nd_covariance_matrix(logits[..., 2:5])
    return xi, omega, alpha


def sample_skew_normal(xi, Omega, alpha, shape=torch.Size(), normalized=True):
    '''
    Args:
        shape: tuple
        xi: [..., dim]
        Omega: [..., dim, dim]
        alpha: [..., dim]

    Return:
        sample: [shape + xi.shape, ..., dim]
    '''
    # create augmented covariance matrix
    dim = int(xi.shape[-1])
    alpha = alpha.unsqueeze(-1)
    omega = batch_diag(Omega).sqrt()
    inv_omega_diag_mat = torch.diag_embed(1. / omega)
    Omega_bar = inv_omega_diag_mat @ Omega @ inv_omega_diag_mat
    scale = 1 / torch.sqrt(1 + alpha.transpose(-1, -2).contiguous() @ Omega_bar @ alpha)
    delta = scale * Omega_bar @ alpha
    omega_aug = torch.cat([delta, Omega_bar], dim=-1)
    delta_aug = torch.cat([torch.ones(*xi.shape[:-1], 1, 1, device=xi.device), delta.transpose(-1, -2).contiguous()], dim=-1)
    omega_aug = torch.cat([delta_aug, omega_aug], dim=-2)
    noise = torch.randn(*(shape + xi.shape[:-1]), dim + 1).to(xi.device)
    omega_sqrt = cholesky_with_exception(omega_aug)
    noise = (omega_sqrt @ noise.unsqueeze(-1)).squeeze(-1)
    x0 = noise[..., 0]
    x = noise[..., 1:]
    z = torch.zeros_like(x)
    indicator = x0 > 0
    z[indicator] = x[indicator]
    z[~indicator] = - x[~indicator]
    xi = left_expand(xi, len(shape))
    omega = left_expand(omega, len(shape))
    sample = xi + z * omega
    return sample


'''
General Skew Normal
'''
def zeta_1(x):
    return univariate_standard_normal_pdf(x) / cdf_normal(x)


def zeta_2(x):
    return - zeta_1(x) ** 2 - x * zeta_1(x)


def delta_of_skew_normal(Omega, alpha):
    '''
    Args:
        Omega: [..., dim, dim]
        alpha: [..., dim]
    '''
    Omega_bar = get_Omega_bar(Omega)
    scale = 1 / torch.sqrt(1 + compute_quadratic_form(alpha, Omega_bar, alpha))
    delta = scale.unsqueeze(-1) * (Omega_bar @ alpha.unsqueeze(-1)).squeeze(-1)
    return delta


def muz_of_skew_normal(Omega, alpha):
    '''
    Args:
        Omega: [..., dim, dim]
        alpha: [..., dim]
    '''
    delta = delta_of_skew_normal(Omega, alpha)
    b = math.sqrt(2 / math.pi)
    return b * delta


def expectation_of_skew_normal(xi, Omega, alpha):
    '''
    Args:
        xi: [..., dim]
        Omega: [..., dim, dim]
        alpha: [..., dim]
    '''
    muz = muz_of_skew_normal(Omega, alpha)
    omega_sqrt = batch_diag(Omega).sqrt().unsqueeze(-1)
    return xi + omega_sqrt * muz


def Sigmaz_of_skew_normal(Omega, alpha):
    '''
    Args:
        Omega: [..., dim, dim]
        alpha: [..., dim]
    '''
    muz = muz_of_skew_normal(Omega, alpha).unsqueeze(-1)
    omega_sqrt = batch_diag(Omega).sqrt()
    omega_sqrt_inv = torch.diag_embed(1. / omega_sqrt)
    omega_bar = omega_sqrt_inv @ Omega @ omega_sqrt_inv
    return omega_bar - muz @ muz.transpose(-1, -2).contiguous()


def covariance_of_skew_normal(Omega, alpha):
    '''
    Args:
        Omega: [..., dim, dim]
        alpha: [..., dim]
    '''
    sigmaz = Sigmaz_of_skew_normal(Omega, alpha)
    omega_sqrt = torch.diag_embed(batch_diag(Omega).sqrt())
    return omega_sqrt @ sigmaz @ omega_sqrt


def mean_of_skew_normal(xi, Omega, alpha):
    '''
    Args:
        xi: [..., dim]
        Omega: [..., dim, dim]
        alpha: [..., dim]
    '''
    muz = muz_of_skew_normal(Omega, alpha)
    omega = batch_diag(Omega).sqrt()
    return xi + omega * muz


def approximate_m0_univariate_skew_normal(alpha):
    '''
    Args:
        mu: [..., 1]
        omega: [..., 1]
        alpha: [..., 1]
    '''
    delta = alpha / (1 + alpha ** 2).sqrt()
    b = math.sqrt(2 / math.pi)
    muz = b * delta
    sigmaz = 1 - muz ** 2
    gamma1 = (4 - math.pi) / 2 * (muz / sigmaz) ** 3
    m0 = muz - (gamma1 * sigmaz) / 2 - torch.sign(alpha) / 2 * (- 2 * math.pi / alpha.abs()).exp()
    return m0


def alpha_only_univariate_skew_normal_nll(x, alpha):
    phi_x = np_univariate_standard_normal_pdf(x)
    Phi_alpha_x = scipy.special.ndtr(alpha * x)
    nll = - np.log(2 * phi_x * Phi_alpha_x)
    return nll


def scipy_m0_univariate_skew_normal(alpha):
    func = partial(alpha_only_univariate_skew_normal_nll, alpha=alpha)
    # delta = alpha / math.sqrt(1 + alpha ** 2)
    # mean = math.sqrt(2 / math.pi) * delta
    # epsilon = abs(mean) * 0.5
    # res = scipy.optimize.minimize_scalar(func, method='brent', bracket=(mean - epsilon, mean + epsilon))
    res = scipy.optimize.minimize_scalar(func, method='brent')
    return res.x


def mode_of_skew_normal(xi, Omega, alpha, approx=True):
    '''
    BUG: PROBLEMATIC
    Args
    Args:
        xi: [..., dim]
        Omega: [..., dim, dim]
        alpha: [..., dim]
    '''
    delta = delta_of_skew_normal(Omega, alpha)
    Omega_bar = get_Omega_bar(Omega)
    inv_Omega_bar = torch.linalg.inv(Omega_bar)
    delta_star = compute_quadratic_form(delta, inv_Omega_bar, delta).sqrt()
    omega = batch_diag(Omega).sqrt()
    # BUG: m0_star has no closed-form expression
    alpha_star = compute_quadratic_form(alpha, Omega_bar, alpha).sqrt()
    if approx:
        # Keep gradient, fast
        m0_star = approximate_m0_univariate_skew_normal(alpha_star)
    else:
        # Break gradient, relatively slow
        if alpha_star.numel() == 1:
            m0_star = scipy_m0_univariate_skew_normal(alpha_star.item())
            m0_star = torch.tensor(m0_star, device=alpha_star.device).float()
        else:
            m0_star = pool_map(scipy_m0_univariate_skew_normal, alpha_star.flatten().detach().cpu().tolist(), jobs=8)
            m0_star = torch.tensor(m0_star, device=alpha_star.device).float().reshape(alpha_star.shape)
    return xi + (m0_star / delta_star).unsqueeze(-1) * omega * delta


def skewness_of_skew_normal(Omega, alpha):
    '''
    Args:
        Omega: [..., dim, dim]
        alpha: [..., dim]
    '''
    muz = muz_of_skew_normal(Omega, alpha)
    sigmaz = Sigmaz_of_skew_normal(Omega, alpha)
    if int(Omega.shape[-1]) == 2:
        inv_sigmaz = inv_2x2_matrix(sigmaz)
    else:
        inv_sigmaz = torch.linalg.inv(sigmaz)
    inner_product = compute_quadratic_form(muz, inv_sigmaz, muz)
    return ((4 - math.pi) / 2) ** 2 * inner_product ** 3


def kurtosis_of_skew_normal(Omega, alpha):
    '''
    Args:
        Omega: [..., dim, dim]
        alpha: [..., dim]
    '''
    muz = muz_of_skew_normal(Omega, alpha)
    sigmaz = Sigmaz_of_skew_normal(Omega, alpha)
    if int(Omega.shape[-1]) == 2:
        inv_sigmaz = inv_2x2_matrix(sigmaz)
    else:
        inv_sigmaz = torch.linalg.inv(sigmaz)
    inner_product = compute_quadratic_form(muz, inv_sigmaz, muz)
    return 2 * (math.pi - 3) * inner_product ** 2


def torch_Mardia_skewness(x):
    '''
    Args:
        x: [..., N, dim]
    '''
    mean = x.mean(-2, keepdim=True)
    delta = x - mean
    # delta: [..., N, dim, 1]
    delta = delta.unsqueeze(-1)
    # covariance: [..., N, dim, dim]
    covariance = delta @ delta.transpose(-1, -2)
    covariance = covariance.mean(-3)
    inv_covariance = torch.linalg.pinv(covariance)
    # delta: [..., N, dim]
    delta = delta.squeeze(-1)
    # skewness: [..., N, N]
    skewness = ((delta @ inv_covariance @ delta.transpose(-1, -2)) ** 3).mean((-1, -2))
    return skewness


def np_Mardia_skewness(x):
    '''
    Args:
        x: [..., N, dim]
    '''
    mean = x.mean(-2, keepdims=True)
    delta = x - mean
    # delta: [..., N, dim, 1]
    delta = delta[..., None]
    # covariance: [..., N, dim, dim]
    covariance = delta @ delta.swapaxes(-1, -2)
    covariance = covariance.mean(-3)
    inv_covariance = np.linalg.inv(covariance)
    # delta: [..., N, dim]
    delta = delta.squeeze(-1)
    # skewness: [..., N, N]
    skewness = ((delta @ inv_covariance @ delta.swapaxes(-1, -2)) ** 3).mean((-1, -2))
    return skewness


'''
cdf and pdf
'''
def np_univariate_standard_normal_pdf(x):
    return np.exp(- x ** 2 / 2) / math.sqrt(2 * math.pi)


def univariate_standard_normal_pdf(x):
    return (- x ** 2 / 2).exp() / math.sqrt(2 * math.pi)


def approx_cdf_normal(x):
    return 1 / 2 * (1 + (math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)).tanh())


def cdf_normal(x):
    return (1 + torch.erf(x / math.sqrt(2))) / 2


def normal_pdf(x, mu, Sigma):
    '''
    Args:
        x: [..., dim]
        mu: [..., dim]
        Sigma: [..., dim, dim]
    '''
    delta = x - mu
    dim = Sigma.size()[-1]
    if dim == 2:
        inv_Sigma = inv_2x2_matrix(Sigma)
        det_Sigma = det_2x2_matrix(Sigma)
    else:
        inv_Sigma = torch.linalg.inv(Sigma)
        det_Sigma = torch.linalg.det(Sigma)
    quadratic_form = compute_quadratic_form(delta, inv_Sigma, delta)
    constant = (2 * math.pi) ** dim
    pdf = (- quadratic_form / 2).exp() / (det_Sigma * constant).sqrt()
    return pdf


def inv_cdf_normal(x):
    return math.sqrt(2) * torch.erfinv(2 * x - 1)


def sample_normal(mu, sigma, shape=torch.Size()):
    '''
    Args:
        mu: [..., dim]
        sigma: [..., dim]
        shape: tuple
    '''
    dim = int(mu.shape[-1])
    noise = torch.randn(*(shape + mu.shape[:-1]), dim).to(mu.device)
    sigma_sqrt = cholesky_with_exception(sigma)
    # sigma_sqrt = torch.linalg.cholesky(sigma)
    sigma_sqrt = left_expand(sigma_sqrt, len(shape))
    noise = (sigma_sqrt @ noise.unsqueeze(-1)).squeeze(-1)
    mu = left_expand(mu, len(shape))
    return mu + noise
