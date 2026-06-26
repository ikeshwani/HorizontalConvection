#!/bin/bash
#SBATCH --job-name="animation"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-node=0                  #this is 0 if cpu run
#SBATCH --cpus-per-task=4
#SBATCH --account=bfxn-delta-cpu
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output="/work/hdd/bfxn/ikeshwani/HorizontalConvection/analysis_scripts/output_message/make_animation%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"

module purge

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/make_animation.jl
