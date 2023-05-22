# import GPy
import numpy as np
import torch
import gpytorch
import sys
from .test_problems import rescale
from .test_problems import Scalarise
import time

class Gpytorch:
    
    def __init__(self, X, y, device):
        self.X_scaler = Scalarise(X)
        self.y_scaler = Scalarise(y)
        self.device = torch.device(device)
        self.X = torch.from_numpy(np.float32(self.X_scaler.arr_scaled)).to(self.device)
        self.y = torch.from_numpy(np.float32(self.y_scaler.arr_scaled.reshape(-1))).to(self.device)
        initial_lengthscales = np.random.uniform(0, 1, self.X.shape[1])
        self.__likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.__model = ExactGPModel(self.X, self.y, self.__likelihood).to(self.device)
        self.__model.covar_module.base_kernel.lengthscale = initial_lengthscales

        self._fit()


    def _predict(self, X):
        # self.__model.to(self.device)
        # self.__likelihood.to(self.device)
        X = self.X_scaler.scale(X)#.reshape(1,-1))
        X = torch.from_numpy(np.float32(X)).to(self.device)#.reshape(-1,X.shape[0])
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.__likelihood(self.__model(X))

        mean = rescale(np.asarray(pred.mean), self.y_scaler.scaler)
        var = pred.variance.numpy() * (self.y_scaler.std_dev**2)
        std_dev = np.sqrt(var)

        return mean, std_dev




class ExactGPModel(gpytorch.models.ExactGP):
    # Takes training data and a likelihood to construct GP
    def __init__(self, X_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(X_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=X_train.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

## model.get_fantasy_model(new_x, new_y)