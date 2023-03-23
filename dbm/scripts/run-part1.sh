#!/bin/bash

##################################################
# RUN DBM (PART 1)
# * pre-run     convert Nifti files to MINC and create input list for pipeline
# * run         run MINC pipeline
##################################################

# settings (change or comment out as needed)
# required
DPATH_PIPELINE="/ipl/quarantine/experimental/2013-02-15"
DPATH_TEMPLATE="/ipl/quarantine/models/icbm152_model_09c"
# TEMPLATE="mni_icbm152_t1_tal_nlin_sym_09c"
TEMPLATE="mni_icbm152_t1_tal_nlin_asym_09c"
# optional
FLAG_SGE_QUEUE="--queue origami.q"
FLAG_OVERWRITE="--overwrite"
# FLAG_DRY_RUN="--dry-run"
# FLAG_INPUT="--minc-input-dir input"
FLAG_OUTPUT="--output-dir output-asym"

FPATH_DOTENV=".env" # relative to current directory

if [ $# -gt 1 ]
then
    echo "Usage: $0 [DATASET_TAG]"
    exit 1
else
    DATASET_TAG=$1
    if [ ! -z $DATASET_TAG ]
    then
        DATASET_TAG_FLAG="--tag $DATASET_TAG"
    fi
fi

# move to scripts directory
DPATH_INITIAL=`pwd`
cd `dirname $(realpath $0)`

# make sure env file exists
if [ ! -f $FPATH_DOTENV ]
then
	echo "File not found: $FPATH_DOTENV. Did you run the init script?"
    exit 2
fi
source $FPATH_DOTENV

# Nifti-to-MINC conversion
# NOTE: --overwrite only overwrites the output list.csv file, not the MINC files
COMMAND_PRE_RUN=" \
    $FPATH_MRI_CODE pre-run \
        $DPATH_OUT_DBM \
        $DATASET_TAG_FLAG \
        $FLAG_INPUT \
        $FLAG_OVERWRITE \
        $FLAG_DRY_RUN \
"
echo $COMMAND_PRE_RUN
eval $COMMAND_PRE_RUN

# MINC DBM pipeline
if [ ! -z "$FLAG_SGE_QUEUE" ]
then
    FLAG_SGE_QUEUE="--sge $FLAG_SGE_QUEUE"
fi

COMMAND_RUN=" \
    $FPATH_MRI_CODE run \
    $DPATH_OUT_DBM \
    $DATASET_TAG_FLAG \
    --pipeline-dir $DPATH_PIPELINE \
    --template-dir $DPATH_TEMPLATE \
    --template $TEMPLATE \
    $FLAG_SGE_QUEUE \
    $FLAG_OUTPUT \
    $FLAG_DRY_RUN \
"
echo $COMMAND_RUN
eval $COMMAND_RUN

# return to original directory
cd $DPATH_INITIAL
