#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=basic_job
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=00-06:00:00
#SBATCH --mail-type=ALL

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

cd /vast/palmer/pi/krishnaswamy_smita/bsw38/GraphML/ZINC
/home/bsw38/.conda/envs/zinc_testing/bin/python -u main.py