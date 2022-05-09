#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH -N 1 --ntasks-per-node=16
#SBATCH --ntasks=1
#SBATCH -t 00:30:00
#SBATCH --mail-user=t.j.loos@student.vu.nl

source /home/${USER}/.bashrc
source activate scaling_rgcn

cd /home/${USER}/RGCN_MscThesis_TiddoLoos

#Run Program
python main.py -dataset AIFB -exp attention

#Copy output data from scratch to home
cp -r "$TMPDIR"/output_dir $HOME
