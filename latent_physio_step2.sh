#!/bin/sh

python train_aux2.py \
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
	--step_size 0.1 \
	--alpha 0.01 \
	--m 100 \
        --load '59645' \
        --load_aux '59645_91853' \
	--gpu 2
