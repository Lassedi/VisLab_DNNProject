#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 05:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=18
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem=120G

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lasse-d-z@web.de

. ~/.bashrc # This line is super important, otherwise the conda activate doesn't work
echo "Start job"

#loading the modules
module load 2022
module load Anaconda3/2022.05
conda activate DNN_project

echo "conda env activated"

#coping the data
cp -r $HOME/FCN_project/Data "$TMPDIR"
cp $HOME/FCN_project/main_snellius.py "$TMPDIR"

echo "Starting Training"

#running the script
python FCN_project/main_snellius.py --inp_dir "$TMPDIR"
echo "Finished training"

#coping the results to local machine
cp -r "$TMPDIR"/model $HOME/Snellius_model
cp -r "$TMPDIR"/runs $HOME/Snellius_runs

echo "Finished output aggregation & script"