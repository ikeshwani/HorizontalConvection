#!/bin/bash

#SBATCH --job-name="hc-ra1e9-chebs"             ## Name of the job.
#SBATCH --output="output_message/hc_chebs_ra1e9.%j.out"             ##output file
#SBATCH --partition=gpuA40x4
#SBATCH --mem=36G
#SBATCH --nodes=1                 ## (-N) number of nodes to use
#SBATCH --ntasks-per-node=1                ## (-n) number of tasks to launch
#SBATCH --cpus-per-task=4
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1                  #this is 0 if cpu run
#SBATCH --gpu-bind=closest
#SBATCH --account=bfxn-delta-gpu         ## my account name #change to cpu or gpu depending 
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"
#SBATCH --exclusive                     #only needed when running on multi-gpu
#SBATCH --no-requeue
#SBATCH -t 48:00:00

rundir1=/work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts/run_case1.jl
rundir2=/work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts/run_case2.jl
rundir3=/work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts/run_case3.jl

#launch 3 parallel julia processes on GPU 0 
#each process generates a separate output log in the output_message dir

module purge
module load cudatoolkit

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts

CUDA_VISIBLE_DEVICES=0 julia --project=../ $rundir1 > output_message/message.run1.${SLURM_JOB_ID}.${SLURM_NODELIST}.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 julia --project=../ $rundir2 > output_message/message.run2.${SLURM_JOB_ID}.${SLURM_NODELIST}.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 julia --project=../ $rundir3 > output_message/message.run3.${SLURM_JOB_ID}.${SLURM_NODELIST}.log 2>&1 &

wait
tail -n 50 run*.log
echo "all runs completed"
date