#!/bin/bash
#SBATCH --partition=normal
#SBATCH --constraint=gold_6130
#SBATCH --job-name=kbisim_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t.j.loos@student.vu.nl

module load 2021

cd $TMPDIR
scp -r ~/scala_setup/fluid_spark_t .
cd $TMPDIR/fluid_spark_t

#Run Program
sbt "runMain BisimTestPipeline"

scp ./exports/* $HOME 
scp ./results/* $HOME
