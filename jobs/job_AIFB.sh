#!/bin/bash
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=AIFB_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 1:00:00
#SBATCH --mail-type=,END
#SBATCH --mail-user=t.j.loos@student.vu.nl

source /home/${USER}/.bashrc
source activate scaling_rgcn

cd "$TMPDIR"/
mkdir ./results
scp -r $HOME/graphdata ./

#Run Program
python /home/loost/RGCN_MscThesis_TiddoLoos/main.py -dataset AIFB

#Copy output directory from scratch to results folder on local machine
cd "$TMPDIR"/
scp results/* $HOME/results/AIFB
