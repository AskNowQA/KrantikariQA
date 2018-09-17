#!/usr/bin/env bash
echo decomposable attention
CUDA_VISIBLE_DEVICES=3 python corechain.py -device cuda -pointwise False -dataset lcquad -model decomposable_attention  &> output/decomposable_attention.txt &
wait
echo bilstm_dot
CUDA_VISIBLE_DEVICES=3 python corechain.py -device cuda -pointwise False -dataset lcquad -model bilstm_dot  &> output/bilstm_dot.txt &
wait
echo bilstm_densedot
CUDA_VISIBLE_DEVICES=3 python corechain.py -device cuda -pointwise False -dataset lcquad -model bilstm_densedot  &> output/bilstm_densedot.txt &
wait
echo bilstm_dense
CUDA_VISIBLE_DEVICES=3 python corechain.py -device cuda -pointwise False -dataset lcquad -model bilstm_dense  &> output/bilstm_dense.txt &
wait