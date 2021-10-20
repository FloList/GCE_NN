#!/bin/bash

#PBS -P u95
#PBS -l walltime=00:14:59
#PBS -N GCE_train
#PBS -l mem=32GB          
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta                     

# Load modules manually (CUDA, cudnn, nccl are loaded with tensorflow)
module purge
module load tensorflow/2.6.0


cd $PBS_O_WORKDIR

export PATH=/scratch/u95/fl9575/GCE_reimplementation/gce_venv/bin:/apps/python3/3.9.2/bin:/opt/pbs/default/bin:/opt/nci/bin:/opt/bin:/opt/Modules/v4.3.0/bin:/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/562/fl9575/.local/bin

export PYTHONPATH=/apps/tensorflow/2.6.0/lib/python3.9/site-packages:/scratch/u95/fl9575/GCE_reimplementation/gce_venv/lib/python3.9/site-packages/

export LD_LIBRARY_PATH=/apps/openmpi/4.1.1/lib:/apps/openmpi/4.1.1/lib/profilers:/apps/nccl/2.10.3-cuda11.4/lib64:/apps/cudnn/8.2.2-cuda11.4/lib64:/apps/cuda/11.4.1/extras/CUPTI/lib64:/apps/cuda/11.4.1/lib64:/apps/intel-ct/2020.3.304/mkl/lib/intel64:/apps/tensorflow/2.6.0/lib:/apps/python3/3.9.2/lib

python3 ${FILENAME}
