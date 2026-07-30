#!/bin/bash
#SBATCH --job-name="animate3d"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-node=0                  #this is 0 if cpu run
#SBATCH --cpus-per-task=4
#SBATCH --account=bfxn-delta-cpu
#SBATCH --mem=40G
#SBATCH --time=2:30:00
#SBATCH --output="/work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts/analysis_scripts/output_message/animate_3d_buoyancy%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"

##module purge

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/animate_3d_buoyancy.jl