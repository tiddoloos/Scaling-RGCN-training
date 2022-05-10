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

module load 2021
cd $TMPDIR
cp ~/scala_setup/myorientdb.img .
singularity run myorientdb.img

cd /home/${USER}/scala_setup/fluid-spark/
#Run Program
sbt "runMain Main resources/configs/tests/manual-test-1.conf"

#stop image
cd $TMPDIR
singularity instance stop myorientdb.img

