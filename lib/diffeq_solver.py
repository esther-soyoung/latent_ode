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
from lib.torchdiffeq_ import odeint as odeint

from lib.cnf_regularization import RegularizedODEfunc

#####################################################################################################

class DiffeqSolver(nn.Module):
	def __init__(self, input_dim, ode_func, reg_func, method, latents, 
			odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu"), train=True):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.latents = latents		
		self.device = device
		self.train = train

		self.ode_func = ode_func
		self.func = None
		nreg = 0
		if reg_func is not None:
			self.func = RegularizedODEfunc(ode_func, reg_func)
			nreg = 1
		self.nreg = nreg

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol


	def forward(self, first_point, time_steps_to_predict, method, backwards = False, **options):
		"""
		# Decode the trajectory through ODE Solver
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
		n_dims = first_point.size()[-1]
		reg_state = torch.zeros(first_point.size(0)).to(first_point)
		if self.nreg > 0 and self.train:  # regularizer state
			assert self.func is not None, 'regularizer function not given'
			state_t= odeint(self.func,  # ode_func & reg_func
				(first_point, reg_state),
				time_steps_to_predict, 
				rtol=[self.odeint_rtol] + [1e20],
				atol=[self.odeint_atol] + [1e20],
				method=method,
				options=options)
			pred_y = state_t[0].permute(1,2,0,3)  # [3, 50, 2208, 20]
			reg_state = state_t[1].permute(1,0)  # [3, 2208]
		else:
			state_t = odeint(self.ode_func,  # ode_func
				first_point, 
				time_steps_to_predict, 
				rtol=self.odeint_rtol,
				atol=self.odeint_atol,
				method=method,
				options=options)
			pred_y = state_t.permute(1,2,0,3)

		assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y, reg_state

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

