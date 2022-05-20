#!/bin/bash
#SBATCH --partition=normal
#SBATCH --constraint=silver_4110
#SBATCH --job-name=AIFB_run
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
python /home/loost/RGCN_MscThesis_TiddoLoos/run_main_k.py -dataset AIFB -k 1

#Copy output directory from scratch to results folder on local machine
cd "$TMPDIR"/
scp results/* $HOME/results/AIFB
