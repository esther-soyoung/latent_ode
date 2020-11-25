#!/bin/sh

python train_aux3.py \
        --niters 50 \
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
	--step_size 0.1 \
        --load '29643' \
	--random-seed $1 \
	--gpu 2
