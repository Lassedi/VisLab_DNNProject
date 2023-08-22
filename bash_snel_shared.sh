#!/bin/bash

#SBATCH -p gpu
#SBATCH -t 10:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --gpus=2
#SBATCH --mem=240G

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
#mktmp -d -p "/scratch-shared/ldietz
mkdir -d -p "/scratch-shared/tmp.oXiVq5zGCK"/Data

cp -r $HOME/FCN_project/Data/MNIST "/scratch-shared/tmp.oXiVq5zGCK"/Data
cp -r $HOME/FCN_project/custom "/scratch-shared/tmp.oXiVq5zGCK"
cp $HOME/FCN_project/main_snellius.py "/scratch-shared/tmp.oXiVq5zGCK"
cd "/scratch-shared/tmp.oXiVq5zGCK"

echo "Starting Training"

#running the script
python main_snellius.py --inp_dir "/scratch-shared/tmp.oXiVq5zGCK" --nepoch 5
echo "Finished training"

#coping the results to local machine
cp -r "$TMPDIR"/model $HOME/Snellius_model
cp -r "$TMPDIR"/runs $HOME/Snellius_runs

echo "Finished output aggregation & script"