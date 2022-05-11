#!/bin/bash
#SBATCH --job-name=AIFB_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=t.j.loos@student.vu.nl

source /home/${USER}/.bashrc
source activate scaling_rgcn

cd "$TMPDIR"/
#Create output directory on scratch
mkdir ./output_dir

#Run Program
python /home/${USER}/RGCN_MscThesis_TiddoLoos/main.py -dataset AIFB

#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME
