#!/bin/bash

##################################################
# RUN DBM (PART 2)
# * post-run     Mask DBM result file and convert to Nifti format
##################################################

# settings (change or comment out as needed)
# required
DPATH_TEMPLATE="/ipl/quarantine/models/icbm152_model_09c"
TEMPLATE="mni_icbm152_t1_tal_nlin_sym_09c"
# optional
FLAG_OVERWRITE="--overwrite"
# FLAG_DRY_RUN="--dry-run"
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

# post-processing of DBM result files
COMMAND_POST_RUN=" \
    $FPATH_MRI_CODE post-run \
        $DPATH_OUT_DBM \
        --template-dir $DPATH_TEMPLATE \
        --template $TEMPLATE \
        $DATASET_TAG_FLAG \
        $FLAG_OVERWRITE \
        $FLAG_DRY_RUN \
"
echo $COMMAND_POST_RUN
eval $COMMAND_POST_RUN

# return to original directory
cd $DPATH_INITIAL
