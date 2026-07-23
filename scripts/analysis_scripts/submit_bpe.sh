#!/bin/bash
#SBATCH --job-name="bpe_ctrl"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=bfxn-delta-cpu
#SBATCH --mem=140G
#SBATCH --time=04:00:00
#SBATCH --output="analysis_scripts/output_message/bpe_control_%j.out"
#SBATCH --mail-user=ikeshwan@uci.edu
#SBATCH --mail-type="BEGIN,END"

# BPE analysis for the flat (control) GRC run, segments 1:15.
# BPEcalc.jl is set to experiment = "control" at the top of the file.

module purge

cd /work/hdd/bfxn/ikeshwani/HorizontalConvection/scripts
julia --project=../ analysis_scripts/BPEcalc.jl
