#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=7g
#SBATCH --out=outfiles/ensemble4.out
#SBATCH -t 48:00:00

source activate pytorch_p37
cd /home/ianpan/ufrc/bengali/src/
/home/ianpan/anaconda3/envs/pytorch_p37/bin/python run.py configs/ensemble_008/i4o0.yaml train --gpu 0 --num-workers 4