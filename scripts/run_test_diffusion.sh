#! /bin/bash
device=$1
python test.py
CUDA_VISIBLE_DEVICES=1 python test_diffusion.py
