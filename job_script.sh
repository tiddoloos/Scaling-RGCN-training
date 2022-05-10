#!/bin/bash
#SBATCH --job-name=test_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu_shared
#SBATCH -t 100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.j.loos@student.vu.nl

module purge all
module load 2021
module load Anaconda3/2021.05

source /home/${USER}/.bashrc
source activate scaling_rgcn

cd /home/${USER}/RGCN_MscThesis_TiddoLoos

#Run Program
python main.py -dataset AM -exp sum
