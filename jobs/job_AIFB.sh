#!/bin/bash
#SBATCH --job-name=AIFB_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 1:00:00
#SBATCH --mail-user=t.j.loos@student.vu.nl

source /home/${USER}/.bashrc
source activate scaling_rgcn

#Create output directory on scratch
cd "$TMPDIR"/
mkdir ./output_dir/AIFB

#Run Program
python /home/loost/RGCN_MscThesis_TiddoLoos/main.py -dataset AIFB

#Copy output directory from scratch to results folder on local machine
scp -r loost@lisa.surfsara.nl:./output_dir/AIFB /Users/tiddo/Documents/Msc\ Artificial\ Intelligence/Thesis_RGCN/RGCN_MscThesis_TiddoLoos/results/
# scp output_dir/* /home/loost/output_dir/AIFB
