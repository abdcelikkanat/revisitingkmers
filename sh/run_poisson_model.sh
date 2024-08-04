#!/usr/bin/bash -l
#SBATCH --job-name=LINEARMODEL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abce@cs.aau.dk

# Define the global variables
MODELNAME=poisson_model
BASEFOLDER=/home/cs.aau.dk/zs74qz/revisitingkmers/
PYTHON="singularity exec --nv ${BASEFOLDER}/../containers/nn python"
SCRIPT_PATH=$BASEFOLDER/src/${MODELNAME}.py

# Model Parameters
INPUT_PATH=${BASEFOLDER}/../kadir/dataset/train_2m.csv
K=4
DIM=256
W=4
EPOCHNUM=1000
LR=0.001
BATCH_SIZE=0
MAXREADNUM=10000
SEED=26042024
POSTFIX=""


# Define the output path
OUTPUT_PATH=${BASEFOLDER}/models/${MODELNAME}_train_2m_k=${K}_d=${DIM}_w=${W}_epoch=${EPOCHNUM}
OUTPUT_PATH=${OUTPUT_PATH}_LR=${LR}_batch=${BATCH_SIZE}_maxread=${MAXREADNUM}_seed=${SEED}${POSTFIX}.model

# Define the command
CMD="$PYTHON ${SCRIPT_PATH} --input ${INPUT_PATH} --k ${K} --dim ${DIM} --w ${W}"
CMD="${CMD} --epoch ${EPOCHNUM} --lr ${LR} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --read_sample_size ${MAXREADNUM} --device cpu --seed ${SEED}"
CMD="${CMD} --output ${OUTPUT_PATH}"

# Run the command
echo ${OUTPUT_PATH}
$CMD

