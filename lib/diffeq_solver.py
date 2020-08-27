###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import time
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from torch.distributions.multivariate_normal import MultivariateNormal

from lib.torchdiffeq_ import odeint_err as odeint_err

from .wrappers.cnf_regularization import RegularizedODEfunc

#####################################################################################################

class DiffeqSolver(nn.Module):
	def __init__(self, input_dim, ode_func, reg_func, method, latents, 
			odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu"), train=True):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.latents = latents		
		self.device = device
		if reg_func is not None:
			ode_func = RegularizedODEfunc(ode_func, reg_func)
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

		self.train = train

	def forward(self, first_point, time_steps_to_predict, backwards = False):
		"""
		# Decode the trajectory through ODE Solver
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
		n_dims = first_point.size()[-1]

		if train:
			pred_y, err = odeint_err(self.ode_func, 
				first_point + reg_states, 
				time_steps_to_predict, 
				rtol=[self.odeint_rtol, self.odeint_rtol] + [1e20] * len(reg_states)
				atol=[self.odeint_atol, self.odeint_atol] + [1e20] * len(reg_states)
				method = self.ode_method)
			pred_y = pred_y.permute(1,2,0,3)
		else:
			pred_y, err = odeint_err(self.ode_func, 
				first_point, 
				time_steps_to_predict, 
				rtol=self.odeint_rtol,
				atol=self.odeint_atol,
				method = self.ode_method)
			pred_y = pred_y.permute(1,2,0,3)

		assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y, err

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, 
		n_traj_samples = 1):
		"""
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		# shape: [n_traj_samples, n_traj, n_tp, n_dim]
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y

