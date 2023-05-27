#!/bin/bash

#SBATCH --job-name=test

#SBATCH --time=10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

python3 ../experiments/pipeline_benchmark.py