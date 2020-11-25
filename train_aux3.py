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
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.nn.functional import relu
import torch.optim as optim

import lib.utils as utils
from lib.plotting import *

from lib.rnn_baselines import *
from lib.ode_rnn import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson
from lib.diffeq_solver import DiffeqSolver
from mujoco_physics import HopperPhysics


# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)
parser.add_argument('--viz', action='store_true', help="Show plots while training")

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
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
parser.add_argument('--step_size', type=float, default=0.03, help="Step size for fixed grid integrators")
parser.add_argument('--alpha', type=float, default=0.05, help="Alpha for aux loss function")
parser.add_argument('--cutoff_coef', type=float, default=1, help="Coefficient of Dopri cutoff value")
parser.add_argument('--m', type=float, default=1000, help="Penalty value M for auxiliary cost")

parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")

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

	IntgExperimentID = int(SystemRandom().random()*100000)
	if not os.path.exists("bosh_experiments/"):
		utils.makedirs("bosh_experiments/")
	intg_ckpt_path = os.path.join('bosh_experiments/', "experiment_" + str(experimentID) + "_" + str(IntgExperimentID) + '.ckpt')


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
			n_labels = data_obj["n_labels"]  #
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

	##################################################################

	#Load checkpoint and evaluate the model
	utils.get_ckpt_model(ckpt_path, model, device)

	log_path = "bosh_logs/" + file_name + "_" + str(experimentID) + "_" + "bosh" + str(IntgExperimentID) + ".log"
	if not os.path.exists("bosh_logs/"):
		utils.makedirs("bosh_logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	## Test ##
	wait_until_kl_inc = 10
	num_batches = data_obj["n_train_batches"]
	itr = num_batches * args.niters

	if itr // num_batches < wait_until_kl_inc:
		kl_coef = 0.
	else:
		kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))

	with torch.no_grad():
		model.train(False)
		dopri_res, fp_enc = compute_loss_all_batches(model,
			data_obj["test_dataloader"], args,
			n_batches = data_obj["n_test_batches"],
			experimentID = experimentID,
			device = device,
			n_traj_samples = 3, kl_coef = kl_coef)
		euler_res, _ = compute_loss_all_batches(model,
			data_obj["test_dataloader"], args,
			n_batches = data_obj["n_test_batches"],
			experimentID = experimentID,
			device = device,
			method = 'euler',
			n_traj_samples = 3, kl_coef = kl_coef)
		rk4_res, _ = compute_loss_all_batches(model,
			data_obj["test_dataloader"], args,
			n_batches = data_obj["n_test_batches"],
			experimentID = experimentID,
			device = device,
			method = 'rk4',
			n_traj_samples = 3, kl_coef = kl_coef)
		bosh3_res, _ = compute_loss_all_batches(model,
			data_obj["test_dataloader"], args,
			n_batches = data_obj["n_test_batches"],
			experimentID = experimentID,
			device = device,
			method = 'bosh3',
			n_traj_samples = 3, kl_coef = kl_coef)

		logger.info("Experiment " + str(experimentID))
		logger.info("Classification AUC (TEST): Dopri5 {:.4f} | Euler {:.4f} | RK4 {:.4f} | Bosh3 {:.4f}".format(dopri_res["auc"], euler_res["auc"], rk4_res["auc"], bosh3_res["auc"]))
		logger.info("NFE:  Dopri5 {:.4f} | Euler {:.4f} | RK4 {:.4f} | Bosh3 {:.4f}".format(dopri_res["nfe"], euler_res["nfe"], rk4_res["nfe"], bosh3_res["nfe"]))
