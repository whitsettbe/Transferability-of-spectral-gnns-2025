#!/bin/bash
#SBATCH --partition=pi_krishnaswamy
#SBATCH --job-name=basic_job
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00-06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=whitsettbe@gmail.com

#################################################
# please DO NOT remove the following two commands
#################################################
module load StdEnv
export SLURM_EXPORT_ENV=ALL
#################################################


# do something
echo "I'm echoing to stdout"
echo "I'm echoing to stderr" 1>&2
echo "My JobID is ${SLURM_JOBID}"
echo "I have ${SLURM_CPUS_ON_NODE} CPUs on node $(hostname -s)"
conda create -n eigvecGCN --clone /vast/palmer/pi/krishnaswamy_smita/hcd22/.conda/envs/eigvecGCN