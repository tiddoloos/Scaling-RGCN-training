#!/bin/bash
#SBATCH --partition=normal
#SBATCH --constraint=silver_4110
#SBATCH --job-name=AM_e_viz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.j.loos@student.vu.nl

source /home/${USER}/.bashrc
source activate scaling_rgcn

cd "$TMPDIR"/
mkdir ./results
mkdir ./results/embeddings

scp -r $HOME/graphdata ./
scp -r $HOME/RGCN_MscThesis_TiddoLoos/baselines ./

#Run Program
python /home/loost/RGCN_MscThesis_TiddoLoos/main.py -dataset AM -i 1 -e_viz True -exp summation -sum bisim

#Copy output directory from scratch to results folder on local machine
cd "$TMPDIR"/
scp -r results/* $HOME/results/AM