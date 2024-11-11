#!/bin/bash
#SBATCH --job-name=llama70bdownload ### Name of the job
#SBATCH --nodes=1                   ### Number of Nodes
#SBATCH --ntasks=1                  ### Number of Tasks
#SBATCH --cpus-per-task=1           ### Number of Tasks per CPU
#SBATCH --gres=gpu:2                ### Number of GPUs, 2 GPUs
#SBATCH --mem=4G                    ### Memory required, 85 GB
#SBATCH --partition=amperenodes     ### Cheaha Partition
#SBATCH --time=02:00:00             ### Estimated Time of Completion, 1 hour
#SBATCH --output=%x_%j.out          ### Slurm Output file, %x is job name, %j is job id
#SBATCH --error=%x_%j.err           ### Slurm Error file, %x is job name, %j is job id

TMPDIR="/local/$USER/$SLURM_JOB_ID"
MY_DATA_DIR=config.save_dir
mkdir -p "$TMPDIR"

# COPY RESEARCH DATA TO LOCAL TEMPORARY DIRECTORY
# Replace $MY_DATA_DIR with the path to your data folder
cp -r "$MY_DATA_DIR" "$TMPDIR"

# YOUR ORIGINAL WORKFLOW GOES HERE
# be sure to load files from "$TMPDIR"!
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Anaconda3

conda activate envnlp

python downloader.py

# CLEAN UP TEMPORARY DIRECTORY
# WARNING!
# Changing the following line can cause permanent, unintended deletion of important data.
rm -rf "$TMPDIR"