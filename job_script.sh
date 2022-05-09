#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH -N 1 --ntasks-per-node=16
#SBATCH --ntasks=1
#SBATCH -t 00:30:00
#SBATCH --mail-user=t.j.loos@student.vu.nl

cd $/home/loost/RGCN_MscThesis_TiddoLoos
source activate scaling_rgcn

#Run Program
python main.py -dataset AIFB -exp attention
