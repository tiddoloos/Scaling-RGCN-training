#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH -N 1 --ntasks-per-node=16
#SBATCH --partition=gpu_shared
#SBATCH --gpus=1
#SBATCH -t 100:00:00
#SBATCH --mail-user=t.j.loos@student.vu.nl

source /home/${USER}/.bashrc
source activate scaling_rgcn

cd /home/${USER}/RGCN_MscThesis_TiddoLoos

#Run Program
python main.py -dataset AM -exp sum
