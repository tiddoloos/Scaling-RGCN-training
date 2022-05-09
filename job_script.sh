#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH -N 1 --ntasks-per-node=16
#SBATCH --ntasks=1
#SBATCH -t 00:30:00
#SBATCH --gpus=1
#SBATCH --mail-user=t.j.loos@student.vu.nl

cd $/home/loost/RGCN_MscThesis_TiddoLoos
source /home/${USER}/.bashrc
source activate Msc_Thesis

#Run Program
python RGCN_MscThesis_TiddoLoos/main.py -dataset AIFB -exp attention

#Copy output data from scratch to home
cp -r "$TMPDIR"/output_dir $HOME
