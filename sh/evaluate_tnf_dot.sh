#!/usr/bin/bash -l
#SBATCH --job-name=EVALUATION
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=712G
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
MODELNAME="kmerprofile"
METRIC=dot
K=4

# Define the evaluation parameters
SPECIES_LIST=("plant") #("reference" "plant" "marine")
MODELLIST=${MODELNAME}
DATA_DIR=${BASEFOLDER}/../kadir/dataset/
MODEL_PATH=${BASEFOLDER}/models/${MODELNAME}.model

# Define the model name
MODELNAME=${MODELNAME}_train_2m_eval-metric=${METRIC}_k=${K}


for SPECIES in ${SPECIES_LIST[@]}
do
# Define output path
OUTPUT_PATH="${BASEFOLDER}/results/${SPECIES}/${MODELNAME}.txt"

# Define the command
CMD="${PYTHON} ${SCRIPT_PATH} --data_dir ${DATA_DIR} --model_list ${MODELLIST}"
CMD=$CMD" --species ${SPECIES} --test_model_dir ${MODEL_PATH}"
CMD=$CMD" --output ${OUTPUT_PATH} --metric ${METRIC} --k ${K}"
echo $CMD
$CMD

done



