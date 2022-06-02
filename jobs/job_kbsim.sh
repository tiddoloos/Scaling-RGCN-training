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
cp ~/scala_setup/fluid-spark .
cd ./fluid-spark

singularity run --pwd $PWD --writable-tmpfs --bind ./orientdb/databases:/orientdb/databases,./orientdb/backup:/orientdb/backup,./orientdb/config/orientdb-server-config.xml:/orientdb/config/orientdb-server-config.xml  ./myorientdb.sif

#Run Program
sbt "runMain Main resources/configs/tests/manual-test-1.conf"

#stop image
singularity instance stop $TMPDIR/myorientdb.img

