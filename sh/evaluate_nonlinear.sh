#!/usr/bin/bash -l
#SBATCH --job-name=EVALUATION
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abce@cs.aau.dk

# Define the global variables
BASEFOLDER=/home/cs.aau.dk/zs74qz/revisitingkmers/
PYTHON="singularity exec --nv ${BASEFOLDER}/../containers/nn python"
SCRIPT_PATH=${BASEFOLDER}/evaluation/binning.py
RESULTS_FOLDER=${BASEFOLDER}/results

export PYTHONPATH=${PYTHONPATH}:${BASEFOLDER}

# Check if a folder exists or not
if ! [ -d ${RESULTS_FOLDER} ]; then
mkdir ${RESULTS_FOLDER}
fi
if ! [ -d ${RESULTS_FOLDER}/reference ]; then
mkdir ${RESULTS_FOLDER}/reference
fi
if ! [ -d ${RESULTS_FOLDER}/marine ]; then
mkdir ${RESULTS_FOLDER}/marine
fi
if ! [ -d ${RESULTS_FOLDER}/plant ]; then
mkdir ${RESULTS_FOLDER}/plant
fi


# Model Parameters
POSTFIX=""
K=4
DIM=256
EPOCHNUM=300
LR=0.001
NEGSAMPLEPERPOS=200
BATCH_SIZE=10000
MAXREADNUM=100000
MODELNAME="nonlinear"

# Define the model name

MODELNAME=${MODELNAME}_train_2m_k=${K}_d=${DIM}_negsampleperpos=${NEGSAMPLEPERPOS}
MODELNAME=${MODELNAME}_epoch=${EPOCHNUM}_LR=${LR}_batch=${BATCH_SIZE}_maxread=${MAXREADNUM}${POSTFIX}

# Define the evaluation parameters
SPECIES_LIST=("reference") #("reference" "plant" "marine")
MODELLIST=nonlinear
DATA_DIR=${BASEFOLDER}/../kadir/dataset/
MODEL_PATH=${BASEFOLDER}/models/${MODELNAME}.model

for SPECIES in ${SPECIES_LIST[@]}
do
# Define output path
OUTPUT_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELNAME}.txt"

# Define the command
CMD="${PYTHON} ${SCRIPT_PATH} --data_dir ${DATA_DIR} --model_list ${MODELLIST}"
CMD=$CMD" --species ${SPECIES} --test_model_dir ${MODEL_PATH}"
CMD=$CMD" --output ${OUTPUT_PATH}"

$CMD
done



