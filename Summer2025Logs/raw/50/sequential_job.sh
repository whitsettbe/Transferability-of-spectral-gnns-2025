#!/bin/bash
#SBATCH --partition=pi_krishnaswamy
#SBATCH --job-name=basic_job
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00-6:00:00
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

cd /vast/palmer/pi/krishnaswamy_smita/bsw38/GraphML/Transferability-of-spectral-gnns-2025/Benchmark-gnn
/home/bsw38/.conda/envs/benchmark_gnn_2/bin/python -u main_molecules_graph_regression.py --dataset ZINC --seed 41 --config 'configs/molecules_graph_regression_Eigval_ZINC.json'
