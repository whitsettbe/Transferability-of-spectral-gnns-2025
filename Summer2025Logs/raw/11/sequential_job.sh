#!/bin/bash
#SBATCH --partition=pi_krishnaswamy
#SBATCH --job-name=basic_job
#SBATCH --ntasks=1 --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
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

cd /home/bsw38/GraphML/ChebNetGNNs/OGB
/home/bsw38/.conda/envs/benchmark_gnn_2/bin/python -u main_dgl.py --dataset ogbg-molhiv --gnn Spec_filters --filename ../SpecFiltTestMolHIV_11 --dropout 0.75 --num_layer 1 --hidden_dim 25 --no_residual --filter_grouping features --l1_reg 0.1
