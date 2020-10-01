###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt
import sklearn as sk

import time
import datetime
import argparse
import numpy as np
import pandas as pd
import random
from random import SystemRandom
from sklearn import model_selection

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils
from lib.plotting import *

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.auxiliary_network import AuxiliaryBlock
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver
from mujoco_physics import HopperPhysics

from lib.utils import compute_loss_all_batches

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default='81209', help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='physionet', help="Dataset to load. Available: physionet, activity, hopper, periodic")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset."
	"Value 1 means quantization by 1 hour, value 0.1 means quantization by 0.1 hour = 6 min")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='rnn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=20, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=40, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=3, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=3, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=50, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=50, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--poisson', action='store_true', help="Model poisson-process likelihood for the density of events in addition to reconstruction.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for Physionet dataset for hospiral mortality")

parser.add_argument('--linear-classif', action='store_true', help="If using a classifier, use a linear classifier instead of 1-layer NN")
parser.add_argument('--extrap', action='store_true', help="Set extrapolation mode. If this flag is not set, run interpolation mode.")

parser.add_argument('-t', '--timepoints', type=int, default=100, help="Total number of time-points")
parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")

parser.add_argument('--gpu', type=int, default=0, help="cuda:")
parser.add_argument('--reg_dopri', type=float, default=0, help="Lambda for Dopri error regularizer.")
parser.add_argument('--reg_kinetic', type=float, default=0, help="Lambda for Kinetic energy regularizer.")
parser.add_argument('--reg_l1', type=float, default=0, help="Lambda for L1 regularizer.")
parser.add_argument('--reg_l2', type=float, default=0, help="Lambda for L2 regularizer(weight decay).")

parser.add_argument('--method', type=str, default='dopri5_err', help="Integrator method: euler, rk4, dopri5_err")
parser.add_argument('--step_size', type=float, default=0.1, help="Step size for fixed grid integrators")
parser.add_argument('--alpha', type=float, default=0.01, help="Alpha for aux loss function")
parser.add_argument('--cutoff_coef', type=float, default=1, help="Coefficient of Dopri cutoff value")

args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################

if __name__ == '__main__':
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed) # if use multi-GPU
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	random.seed(args.random_seed)

	experimentID = args.load
	if experimentID is None:
		raise Exception("Please provide experiment ID to load")
	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

	AUXexperimentID = int(SystemRandom().random()*100000)
	if not os.path.exists("aux_experiments/"):
		utils.makedirs("aux_experiments/")
	aux_ckpt_path = os.path.join('aux_experiments/', "experiment_" + str(experimentID) + "_" + str(AUXexperimentID) + '.ckpt')

	start = time.time()
	print("Sampling dataset of {} training examples".format(args.n))
	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)

	utils.makedirs("results/")

	##################################################################
	data_obj = parse_datasets(args, device)
	input_dim = data_obj["input_dim"]  # 41
	classif_per_tp = False
	if ("classif_per_tp" in data_obj):
		# do classification per time point rather than on a time series as a whole
		classif_per_tp = data_obj["classif_per_tp"]

	n_labels = 1
	if args.classif:
		if ("n_labels" in data_obj):
			n_labels = data_obj["n_labels"]
		else:
			raise Exception("Please provide number of labels for classification task")

	##################################################################
	# Create the model
	obsrv_std = 0.01
	obsrv_std = torch.Tensor([obsrv_std]).to(device)

	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

	if args.latent_ode:
		model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
			classif_per_tp = classif_per_tp,
			n_labels = n_labels)
	else:
		raise Exception("Model not specified")

	##### Auxiliary Network #####
	n_intg = 3  # [dopri, euler, rk4] 
	aux_layers = [AuxiliaryBlock(model.latent_dim, n_intg)]
	aux_net = nn.Sequential(*aux_layers).to(device)
	aux_opt = optim.Adamax(aux_net.parameters(), lr=args.lr)

	##################################################################

	if args.viz:
		viz = Visualizations(device)

	##################################################################
	
	#Load checkpoint and evaluate the model
	utils.get_ckpt_model(ckpt_path, model, device)

	##################################################################
	# Training

	log_path = "aux_logs/" + file_name + "_" + str(experimentID) + "_" + str(AUXexperimentID) + ".log"
	if not os.path.exists("aux_logs/"):
		utils.makedirs("aux_logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	# aux_criterion = nn.NLLLoss().cuda()
	aux_criterion = nn.MSELoss().to(device)

	num_batches = data_obj["n_train_batches"]  # 64

	##### Train Aux Net #####
	logger.info("### Auxiliary Network : step size {}###".format(args.step_size))
	for itr in range(1, num_batches * (args.niters + 1)):  # 100
		print('Iter: ' + str(itr))
		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])
		#### Dynamic Integrators #####
		# dict_keys(['observed_data', 'observed_tp', 'data_to_predict', 'tp_to_predict', 
		# 'observed_mask', 'mask_predicted_data', 'labels', 'mode'])
		with torch.no_grad():
			dopri_res, fp_enc, cutoff = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
			euler_res, _, _ = model.compute_all_losses(batch_dict, method='euler', cut_off=cutoff*args.cutoff_coef, n_traj_samples=3, kl_coef=kl_coef)
			rk4_res, _, _ = model.compute_all_losses(batch_dict, method='rk4', cut_off=cutoff*args.cutoff_coef, n_traj_samples=3, kl_coef=kl_coef)
		##############################

		##### Auxiliary Network #####
		n_traj_samples, n_traj, n_dims = fp_enc.size()  # 3, 50, 20

		dopri_intg = torch.tensor([1, 0, 0]).repeat(n_traj_samples * n_traj, 1).type(torch.FloatTensor)  # [150, 3]
		dopri_cost = torch.Tensor(dopri_res['cost']).unsqueeze(-1)  # [150, 1]
		dopri_truth = (dopri_cost * dopri_intg).to(device)  # [150, 3]

		euler_intg = torch.tensor([0, 1, 0]).repeat(n_traj_samples * n_traj, 1).type(torch.FloatTensor)  # [150, 3]
		euler_cost = torch.Tensor(euler_res['cost']).unsqueeze(-1)  # [150, 1]
		euler_truth = (euler_cost * euler_intg).to(device)  # [150, 3]

		rk4_intg = torch.tensor([0, 0, 1]).repeat(n_traj_samples * n_traj, 1).type(torch.FloatTensor)  # [150, 3]
		rk4_cost = torch.Tensor(rk4_res['cost']).unsqueeze(-1)  # [150, 1]
		rk4_truth = (rk4_cost * rk4_intg).to(device)  # [150, 3]

		aux_truth = dopri_truth + euler_truth + rk4_truth  # [150, 3]
		# aux_truth = torch.max(aux_truth, 1)[1]  # [150]

		aux_opt.zero_grad()
		aux_net.train()
		utils.update_learning_rate(aux_opt, decay_rate = 0.999, lowest = args.lr / 10)
		aux_y = aux_net(fp_enc.clone().detach())  # [3, 50, 3]
		aux_y = aux_y.view(-1, n_intg).type(aux_truth.type()) # [150, 3]

		aux_loss = torch.sqrt(aux_criterion(aux_y, aux_truth))
		logger.info("Iter: {} | Train loss (one batch): {}".format(itr, aux_loss.detach()))
		aux_loss.backward()
		aux_opt.step()
		##############################
	##############################

	############ TEST ############
	with torch.no_grad():
		#### Dynamic Integrators #####
		test_dopri, fp_enc = compute_loss_all_batches(model,
			data_obj["test_dataloader"], args,
			n_batches = data_obj["n_test_batches"],
			experimentID = experimentID,
			device = device,
			n_traj_samples = 3, kl_coef = kl_coef)
		test_euler, _ = compute_loss_all_batches(model,
			data_obj["test_dataloader"], args,
			n_batches = data_obj["n_test_batches"],
			experimentID = experimentID,
			device = device,
			method = 'euler',
			n_traj_samples = 3, kl_coef = kl_coef)
		test_rk4, _ = compute_loss_all_batches(model,
			data_obj["test_dataloader"], args,
			n_batches = data_obj["n_test_batches"],
			experimentID = experimentID,
			device = device,
			method = 'rk4',
			n_traj_samples = 3, kl_coef = kl_coef)

		logger.info("Experiment " + str(experimentID))
		##############################

		##### Auxiliary Network #####
		num_batches = data_obj["n_test_batches"]  # 50
		dopri_cnt, euler_cnt, rk4_cnt = 0, 0, 0
		aux_t = 0
		classif_predictions = torch.Tensor([]).to(device)
		all_test_labels =  torch.Tensor([]).to(device)

		logger.info("### Auxiliary Network : n_test_batches {} ###".format(num_batches))
		for itr in range(1, num_batches + 1):
			wait_until_kl_inc = 10
			if itr // num_batches < wait_until_kl_inc:
				kl_coef = 0.
			else:
				kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

			batch_dict = utils.get_next_batch(data_obj["test_dataloader"])
			dopri_res, fp_enc, cutoff = model.compute_all_losses(batch_dict, n_traj_samples = 3, kl_coef = kl_coef)
			euler_res, _, _ = model.compute_all_losses(batch_dict, method='euler', cut_off=cutoff*args.cutoff_coef, n_traj_samples=3, kl_coef=kl_coef)
			rk4_res, _, _ = model.compute_all_losses(batch_dict, method='rk4', cut_off=cutoff*args.cutoff_coef, n_traj_samples=3, kl_coef=kl_coef)

			n_traj_samples, n_traj, n_dims = fp_enc.size()  # 3, 20, 20

			dopri_intg = torch.tensor([1, 0, 0]).repeat(n_traj_samples * n_traj, 1).type(torch.FloatTensor)  # [60, 3]
			dopri_cost = torch.Tensor(dopri_res['cost']).unsqueeze(-1)  # [60, 1]
			dopri_truth = (dopri_cost * dopri_intg).to(device)  # [60, 3]

			euler_intg = torch.tensor([0, 1, 0]).repeat(n_traj_samples * n_traj, 1).type(torch.FloatTensor)  # [60, 3]
			euler_cost = torch.Tensor(euler_res['cost']).unsqueeze(-1)  # [60, 1]
			euler_truth = (euler_cost * euler_intg).to(device)  # [60, 3]

			rk4_intg = torch.tensor([0, 0, 1]).repeat(n_traj_samples * n_traj, 1).type(torch.FloatTensor)  # [60, 3]
			rk4_cost = torch.Tensor(rk4_res['cost']).unsqueeze(-1)  # [60, 1]
			rk4_truth = (rk4_cost * rk4_intg).to(device)  # [60, 3]

			aux_truth = dopri_truth + euler_truth + rk4_truth  # [60, 3]

			aux_net.eval()
			t = time.time()
			aux_y = aux_net(fp_enc)  # [3, 20, 3]
			aux_t += time.time() - t
			aux_y = aux_y.view(-1, n_intg) # [60, 3]
			aux_test_loss = torch.sqrt(aux_criterion(aux_y, aux_truth))

			# Choice of integrator
			aux_y_sum = torch.sum(aux_y, dim=0)  # [3]
			pred_y = torch.min(aux_y_sum, 0)[1].item()  # 0: dopri5, 1: euler, 2: rk4
			results = {}
			if pred_y == 0:
				pred_integrator = 'dopri5'
				dopri_cnt += 1
				results = dopri_res
			elif pred_y == 1:
				pred_integrator = 'euler'
				euler_cnt += 1
				results = euler_res
			else:
				pred_integrator = 'rk4'
				rk4_cnt += 1
				results = rk4_res

			# overall AUC
			classif_predictions = torch.cat((classif_predictions, 
				results["label_predictions"].reshape(n_traj_samples, -1, n_labels)),1)
			all_test_labels = torch.cat((all_test_labels, 
				batch_dict["labels"].reshape(-1, n_labels)),0)

			# Actual costs
			total_cost_dopri = torch.sum(dopri_cost.squeeze().view(-1)).item()
			total_cost_euler = torch.sum(euler_cost.squeeze().view(-1)).item()
			total_cost_rk4 = torch.sum(rk4_cost.squeeze().view(-1)).item()
			min_loss = min(total_cost_dopri, total_cost_euler, total_cost_rk4)
			if min_loss == total_cost_dopri:
				label_integrator = 'dopri5'
			elif min_loss == total_cost_euler:
				label_integrator = 'euler'
			else:
				label_integrator = 'rk4'

			logger.info("Iter {} | Test loss (one batch): {}".format(itr, aux_test_loss))
			logger.info("Cost(alpha {}) for Dopri integrator (one batch): {}".format(args.alpha, total_cost_dopri))
			logger.info("Cost(alpha {}) for Euler integrator (one batch): {}".format(args.alpha, total_cost_euler))
			logger.info("Cost(alpha {}) for RK4 integrator (one batch): {}".format(args.alpha, total_cost_rk4))
			logger.info("Choice of integrator (one batch): {}".format(pred_integrator))
			logger.info("Auxiliary network predicted {}".format(label_integrator == pred_integrator))
			logger.info("AUC of the choice (one batch): {}".format(results['auc']))

			torch.save({
				'args': args,
				'state_dict': aux_net.state_dict(),
			}, aux_ckpt_path)
		##############################

		##### Overall AUC of the Choice #####

		all_test_labels = all_test_labels.repeat(n_traj_samples,1,1)

		idx_not_nan = ~torch.isnan(all_test_labels)
		classif_predictions = classif_predictions[idx_not_nan]
		all_test_labels = all_test_labels[idx_not_nan]

		overall_auc = 0.
		if torch.sum(all_test_labels) != 0.:
			print("Number of labeled examples: {}".format(len(all_test_labels.reshape(-1))))
			print("Number of examples with mortality 1: {}".format(torch.sum(all_test_labels == 1.)))

			# Cannot compute AUC with only 1 class
			overall_auc = sk.metrics.roc_auc_score(all_test_labels.cpu().numpy().reshape(-1), 
				classif_predictions.cpu().numpy().reshape(-1))
		else:
			print("Warning: Couldn't compute AUC -- all examples are from the same class")
		###################################

	############## LOGGER ###############
	logger.info('############### TOTAL ###############')
	logger.info("Classification AUC : Dopri {:.4f} | Euler {:.4f} | RK4 {:.4f}".format(test_dopri["auc"], test_euler["auc"], test_rk4["auc"]))
	logger.info("NFE (average): Dopri {} | Euler {} | RK4 {}".format(test_dopri['nfe'], test_euler['nfe'], test_rk4['nfe']))
	logger.info("Elapsed time (average): Dopri {} | Euler {} | RK4 {}".format(test_dopri['elapsed_time'], test_euler['elapsed_time'], test_rk4['elapsed_time']))
	logger.info('############### Aux Net Average ###############')
	logger.info('AUC of the choice {:.4f} | Choice of Dopri5 {} | Euler {} | RK4 {}'.format(overall_auc, dopri_cnt, euler_cnt, rk4_cnt))
	logger.info('Aux Net Runtime: {:.4f}'.format(t))

	############## SAVE MODEL ###############
	torch.save({
		'args': args,
		'state_dict': aux_net.state_dict(),
	}, aux_ckpt_path)

