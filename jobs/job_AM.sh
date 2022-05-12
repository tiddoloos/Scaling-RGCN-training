#!/bin/bash
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1
#SBATCH --job-name=AM_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 100:00:00
#SBATCH --mail-type=,END
#SBATCH --mail-user=t.j.loos@student.vu.nl

source /home/${USER}/.bashrc
source activate scaling_rgcn

cd "$TMPDIR"/
mkdir ./results
scp -r $HOME/graphdata ./

#Run Program
python /home/loost/RGCN_MscThesis_TiddoLoos/main.py -dataset AM

#Copy output directory from scratch to results folder on local machine
cd "$TMPDIR"/
scp results/* $HOME/results/AM