import tqdm
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from gpytorch.lazy import TriangularLazyTensor, delazify, MatmulLazyTensor
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.errors import CachingError


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SVGP_new(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGP_new, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(delazify(induc_induc_covar).double())
        return TriangularLazyTensor(L)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def loss(self, x, y, likelihood, beta=1.0):
        """
        Compute the loss for the model, which is the negative variational lower bound.
        :param x: Input data, shape (..., num_data, input_dim)
        :param y: Target data, shape (..., num_data,)
        :param beta: Scaling factor for the KL divergence term, default is 1.0
        :return:
        """
        num_data = x.size(-2)  # number of data points in the batch

        # step 1: compute the KL divergence of the variational distribution from the prior
        kl_divergence = self.variational_strategy.kl_divergence().div(num_data / beta)

        # step 2: compute the log likelihood of the data under the model
        # retrieve variational mean:  K_ZZ^{-1/2} \mu_Z, shape (..., num_inducing_points)
        variational_mean = self.variational_strategy.variational_distribution.mean
        # retrieve variational covariance: K_ZZ^{-1/2} S K_ZZ^{-1/2}, shape (..., num_inducing_points, num_inducing_points)
        variational_covar = self.variational_strategy.variational_distribution.lazy_covariance_matrix


        # compute covariance matrices
        induc_induc_covar = self.covar_module(self.variational_strategy.inducing_points).add_jitter()
        induc_data_covar = self.covar_module(self.variational_strategy.inducing_points, x).evaluate()

        L = self._cholesky_factor(induc_induc_covar)
        # compute the cross covariance: K_ZZ^{-1/2} K_ZX, shape (..., num_data, num_inducing_points)
        interp_term = L.inv_matmul(induc_data_covar.double()).to(x.dtype)

        # compute K_XZ K_ZZ^{-1} \mu_Z, shape (..., num_data) todo: note here that we use zero-mean here!!!!
        mean_term = torch.matmul(interp_term.transpose(-1, -2), variational_mean.unsqueeze(-1)).squeeze(-1)

        # compute K_XX, shape (..., num_data, num_data)
        K_XX = self.covar_module(x).add_jitter()
        # compute the covariance term: K_XX - K_XZ K_ZZ^{-1} K_XZ^T, shape (..., num_data, num_data)
        covar_term = K_XX - torch.matmul(interp_term.transpose(-1, -2), interp_term)
        # Add noise to the covariance term
        sigma_noise = likelihood.noise
        covar_term = covar_term + sigma_noise * torch.eye(covar_term.size(-1), device=covar_term.device)

        # Extract diagonal elements (shape: [..., num_data])
        diag_elements = covar_term.diag()
        # Compute log determinant (sum of log diagonals)
        logdet_diag = torch.log(diag_elements).sum(-1)
        # Compute the log likelihood
        cnst = -0.5 * num_data * math.log(2 * math.pi)
        diff_mean = y - mean_term
        log_likelihood = -0.5 * (logdet_diag + torch.sum(diff_mean ** 2 / diag_elements, dim=-1)) + cnst

        # step 3: compute: (K_XZ K_ZZ^{-1} S K_ZZ^{-1} K_ZX) / (2 * (K_XX - K_XZ K_ZZ^{-1} K_XZ^T + sigma_noise^2 I))
        # Compute the covariance term: K_XZ K_ZZ^{-1} S K_ZZ^{-1} K_ZX,  shape (..., num_data)
        nominator = MatmulLazyTensor(interp_term.transpose(-1, -2), variational_covar @ interp_term).diag()
        # Compute the denominator: 2 * (K_XX - K_XZ K_ZZ^{-1} K_XZ^T + sigma_noise^2 I)
        denominator = 2 * diag_elements
        # Compute the final term
        reg_term = torch.sum(nominator / denominator, dim=-1)

        # step 4: Combine the terms to get the loss
        ELBO = -kl_divergence + log_likelihood.div(num_data) - reg_term.div(num_data)
        return -ELBO

    def titsias25_loss(self, x, y, likelihood, beta=1.0):
        """
        Compute the loss for the model, which is the negative variational lower bound.
        :param x: Input data, shape (..., num_data, input_dim)
        :param y: Target data, shape (..., num_data,)
        :param beta: Scaling factor for the KL divergence term, default is 1.0
        :return:
        """
        num_data = x.size(-2)  # number of data points in the batch

        # step 1: compute the KL divergence of the variational distribution from the prior
        kl_divergence = self.variational_strategy.kl_divergence().div(num_data / beta)

        # step 2: compute the log likelihood of the data under the model
        # retrieve variational mean:  K_ZZ^{-1/2} \mu_Z, shape (..., num_inducing_points)
        variational_mean = self.variational_strategy.variational_distribution.mean
        # retrieve variational covariance: K_ZZ^{-1/2} S K_ZZ^{-1/2}, shape (..., num_inducing_points, num_inducing_points)
        variational_covar = self.variational_strategy.variational_distribution.lazy_covariance_matrix


        # compute covariance matrices
        induc_induc_covar = self.covar_module(self.variational_strategy.inducing_points).add_jitter()
        induc_data_covar = self.covar_module(self.variational_strategy.inducing_points, x).evaluate()

        L = self._cholesky_factor(induc_induc_covar)
        # compute the cross covariance: K_ZZ^{-1/2} K_ZX, shape (..., num_data, num_inducing_points)
        interp_term = L.inv_matmul(induc_data_covar.double()).to(x.dtype)

        # compute K_XZ K_ZZ^{-1} \mu_Z, shape (..., num_data) todo: note here that we use zero-mean here!!!!
        mean_term = torch.matmul(interp_term.transpose(-1, -2), variational_mean.unsqueeze(-1)).squeeze(-1)

        # compute K_XX, shape (..., num_data, num_data)
        K_XX = self.covar_module(x).add_jitter()
        # compute the covariance term: K_XX - K_XZ K_ZZ^{-1} K_XZ^T, shape (..., num_data, num_data)
        covar_term = K_XX - torch.matmul(interp_term.transpose(-1, -2), interp_term)
        # Add noise to the covariance term
        sigma_noise = likelihood.noise * torch.ones(covar_term.size(-1), device=covar_term.device)

        # Extract diagonal elements (shape: [..., num_data])
        # Compute log determinant (sum of log diagonals)
        logdet_diag = torch.log(sigma_noise).sum(-1)
        # Compute the log likelihood
        cnst = -0.5 * num_data * math.log(2 * math.pi)
        diff_mean = y - mean_term
        log_likelihood = -0.5 * (logdet_diag + torch.sum(diff_mean ** 2 / likelihood.noise, dim=-1)) + cnst

        # step 3: compute: (K_XZ K_ZZ^{-1} S K_ZZ^{-1} K_ZX) / (2 * (K_XX - K_XZ K_ZZ^{-1} K_XZ^T + sigma_noise^2 I))
        # Compute the covariance term: K_XZ K_ZZ^{-1} S K_ZZ^{-1} K_ZX,  shape (..., num_data)
        nominator = MatmulLazyTensor(interp_term.transpose(-1, -2), variational_covar @ interp_term).diag()
        # Compute the denominator: 2 * (K_XX - K_XZ K_ZZ^{-1} K_XZ^T + sigma_noise^2 I)
        denominator = 2 * likelihood.noise
        # Compute the final term
        reg_term = torch.sum(nominator / denominator, dim=-1)

        # step 4: Compute: 0.5 * log( 1 +  covar_term.diag() / likelihood.noise)
        ratio_term = 0.5 * torch.log(1 + covar_term.diag() / likelihood.noise).sum(-1)

        # step 5: Combine the terms to get the loss
        ELBO = -kl_divergence + log_likelihood.div(num_data) - reg_term.div(num_data) - ratio_term.div(num_data)
        return -ELBO
