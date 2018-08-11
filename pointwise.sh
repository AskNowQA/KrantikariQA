#!/usr/bin/env bash
echo decomposable attention
CUDA_VISIBLE_DEVICES=1 python corechain.py -device cuda -pointwise True -dataset lcquad -model decomposable_attention  &> output/decomposable_attention_1.txt &
wait
echo bilstm_dot
CUDA_VISIBLE_DEVICES=1 python corechain.py -device cuda -pointwise True -dataset lcquad -model bilstm_dot  &> output/bilstm_dot_1.txt &
wait
echo bilstm_densedot
CUDA_VISIBLE_DEVICES=1 python corechain.py -device cuda -pointwise True -dataset lcquad -model bilstm_densedot  &> output/bilstm_densedot.txt &
wait