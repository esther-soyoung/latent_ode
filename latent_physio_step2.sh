#!/bin/sh

python train_aux.py \
        --niters 5 \
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
        --load '5755' \
	--alpha 0.3 \
	--step_size 18 \
        --gpu 3
