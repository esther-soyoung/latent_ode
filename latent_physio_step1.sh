#!/bin/sh

python run_models.py \
        --niters 100 \
        -n 8000 \
        -l 20 \
        --latent-ode \
        --z0-encoder 'rnn' \
        --dataset 'physionet' \
        --rec-dims 40 \
        --rec-layers 3 \
        --gen-layers 3 \
        --units 50 \
        --gru-units 50 \
        --quantization 0.016 \
        --classif \
	--reg_dopri $1 \
	--reg_kinetic $2 \
	--reg_l1 $3 \
	--reg_l2 $4