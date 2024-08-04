#!/usr/bin/bash -l
#SBATCH --job-name=MODEL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abce@cs.aau.dk

nvidia-smi

# Define the global variables
MODELNAME=nonlinear
BASEFOLDER=/home/cs.aau.dk/zs74qz/revisitingkmers/
PYTHON="singularity exec --nv ${BASEFOLDER}/../containers/nn python"
SCRIPT_PATH=${BASEFOLDER}/src/nonlinear.py

# Model Parameters
INPUT_PATH=$BASEFOLDER/../kadir/dataset/train_2m.csv
POSTFIX=""
K=4
DIM=256
EPOCHNUM=300
LR=0.001
NEGSAMPLEPERPOS=200
BATCH_SIZE=10000
MAXREADNUM=100000 
SEED=26042024
CHECKPOINT=0

# Define the output path
OUTPUT_PATH=${BASEFOLDER}/models/${MODELNAME}_train_2m_k=${K}_d=${DIM}_negsampleperpos=${NEGSAMPLEPERPOS}
OUTPUT_PATH=${OUTPUT_PATH}_epoch=${EPOCHNUM}_LR=${LR}_batch=${BATCH_SIZE}_maxread=${MAXREADNUM}_seed=${SEED}${POSTFIX}.model

# Define the command
CMD="$PYTHON ${SCRIPT_PATH} --input $INPUT_PATH --k ${K} --epoch $EPOCHNUM --lr $LR"
CMD="${CMD} --neg_sample_per_pos ${NEGSAMPLEPERPOS} --max_read_num ${MAXREADNUM}"
CMD="${CMD} --batch_size ${BATCH_SIZE} --device cuda --output ${OUTPUT_PATH} --seed ${SEED} --checkpoint ${CHECKPOINT}"

# Run the command
echo ${OUTPUT_PATH}
$CMD
