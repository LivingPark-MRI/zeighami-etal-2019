#!/bin/bash

##################################################
# CHECK RESULTS AVAILABILITY
# * TODO
##################################################

# settings
PATTERN_COHORT="zeighami-etal-2019-cohort-{}.csv"
FNAME_BAD_SCANS="bad_scans.csv" # TODO
# OVERWRITE_FLAG="--overwrite"

FPATH_DOTENV=".env" # relative to current directory

# temporary file
FPATH_TMP=$(mktemp /tmp/check_results_availability.XXXXXX)

# ========================================
# functions
# ========================================
message() {
	if [ $# == 1 ]
	then
		STEP_NAME=$1
		WIDTH=$((${#STEP_NAME} + 4))
		printf "%*s" $WIDTH "" | tr ' ' '-'
		printf "\n| ${STEP_NAME} |\n"
		printf "%*s" $WIDTH "" | tr ' ' '-'
		printf "\n"
	fi
}

exit_if_error() {
	if [ $# != 1 ]
	then
		echo "No argument passed to exit_if_error"
		exit 4
	fi
	EXIT_CODE=$1
	if [ $EXIT_CODE -ne 0 ]
	then
		exit $EXIT_CODE
	fi
}

# ========================================
# parse inputs
# ========================================
if [ $# != 1 ]
then
    echo "Usage: $0 COHORT_ID"
    exit 1
else
    COHORT_ID=$1
    if [ ! -z $COHORT_ID ]
    then
        DATASET_TAG_FLAG="--tag $COHORT_ID"
    fi
fi

# move to scripts directory
DPATH_INITIAL=`pwd`
cd `dirname $(realpath $0)`

# ========================================
# load environment variables
# ========================================

# make sure env file exists
if [ ! -f $FPATH_DOTENV ]
then
	echo "File not found: $FPATH_DOTENV. Did you run the init script?"
    exit 2
fi
source $FPATH_DOTENV

# ========================================
# make sure cohort file exists
# ========================================
FNAME_COHORT="${PATTERN_COHORT/\{\}/$COHORT_ID}"
FPATH_COHORT="$DPATH_ROOT/$FNAME_COHORT"
if [ ! -f $FPATH_COHORT ]
then
	echo "Cohort file not found: $FPATH_COHORT"
	exit 3
fi

# ========================================
# generate BIDS list file
# TODO check if it already exists first?
# ========================================
COMMAND_BIDS_LIST=" \
	$FPATH_MRI_CODE bids-list \
		$DPATH_OUT_DBM \
		$DPATH_BIDS \
"

echo $COMMAND_BIDS_LIST
eval $COMMAND_BIDS_LIST

# ========================================
# filter list of T1 filepaths based on cohort
# ========================================
COMMAND_BIDS_FILTER=" \
	$FPATH_MRI_CODE bids-filter \
		$DPATH_OUT_DBM \
		$FPATH_COHORT \
		$DATASET_TAG_FLAG \
		--overwrite \
	| tee $FPATH_TMP \
"

echo $COMMAND_BIDS_FILTER
eval $COMMAND_BIDS_FILTER

COHORT_ID_BIDS_LIST=$(grep COHORT_ID $FPATH_TMP | awk -F '=' '{ print $2 }' )

if [ $COHORT_ID_BIDS_LIST != $COHORT_ID ]
then
	message "COHORT ID CHANGED TO $COHORT_ID_BIDS_LIST AFTER GENERATING BIDS LIST"
	FPATH_BIDS_LIST_COHORT_TMP=$FPATH_BIDS_LIST_COHORT
	FPATH_BIDS_LIST_COHORT=${FPATH_BIDS_LIST_COHORT/"$COHORT_ID"/"$COHORT_ID_BIDS_LIST"}
	mv -v $FPATH_BIDS_LIST_COHORT_TMP $FPATH_BIDS_LIST_COHORT
	exit_if_error $?
fi

# # TODO delete missing input list

# # ========================================
# # check DBM processing status
# # ========================================
# message "DBM-STATUS"
# $FPATH_MRI_CODE dbm-status \
# 	$FPATH_BIDS_LIST_COHORT \
# 	$DPATH_OUT_DBM \
# 	--step denoised .denoised.mnc \
# 	--step lin_reg .denoised.norm_lr.masked.mnc \
# 	--step lin_reg_mask .denoised.norm_lr_mask.mnc \
# 	--step nonlin_reg .denoised.norm_lr.masked.nlr_level$NLR_LEVEL.mnc \
# 	--step dbm_nii .denoised.norm_lr.masked.nlr_level$NLR_LEVEL.dbm_fwhm$DBM_FWHM.reshaped.masked.nii.gz \
# 	--overwrite
# exit_if_error $?

# # TODO check if new missing input list was written

# message "COPYING"
# FPATH_PROC_STATUS_COHORT="${DPATH_OUT_DBM}/proc_status-${COHORT_ID}.csv"
# cp -v $DPATH_OUT_DBM/proc_status.csv $FPATH_PROC_STATUS_COHORT
# exit_if_error $?

# # ========================================
# # build list of filenames for ICA
# # ========================================
# message "DBM-LIST"
# $FPATH_MRI_CODE dbm-list \
# 	$DPATH_OUT_DBM \
# 	$FPATH_DBM_LIST \
# 	--overwrite | tee $FPATH_TMP
# exit_if_error $?

# COHORT_ID_DBM_LIST=$(grep COHORT_ID ${FPATH_TMP} | awk -F '=' '{ print $2 }' ) \

# if [ $COHORT_ID_DBM_LIST != $COHORT_ID_BIDS_LIST ]
# then
# 	message "COHORT ID CHANGED TO ${COHORT_ID_DBM_LIST} AFTER GENERATING DBM LIST"
# fi

# message "COPYING"
# FPATH_DBM_LIST_COHORT=$DPATH_OUT_ICA/dbm_list-$COHORT_ID_DBM_LIST.txt
# cp -v $FPATH_DBM_LIST $FPATH_DBM_LIST_COHORT
# exit_if_error $?

# # ========================================
# # check if ICA results exist
# # ========================================
# DIM="30"
# # DIMEST="lap" # 'lap', 'bic', 'mdl', 'aic', 'mean'
# # SHUFFLE="y" # shuffles if non-empty
# SEP_SUFFIX="-"
# if [[ ! (-z $DIM || -z $DIMEST) ]]
# then
# 	message "Only one of DIM and DIMEST can be set (got $DIM and $DIMEST)"
# elif [ ! -z $DIM ]
# then
# 	DIM_FLAG="--dim $DIM"
# 	ICA_SUFFIX="${SEP_SUFFIX}${DIM}"
# elif [ ! -z $DIMEST ]
# then
# 	DIM_FLAG="--dimest $DIMEST"
# 	ICA_SUFFIX="${SEP_SUFFIX}${DIMEST}"
# fi
# if [ ! -z $SHUFFLE ]
# then
# 	ICA_SUFFIX="${ICA_SUFFIX}_shuffle"
# 	SHUFFLE_FLAG='--shuffle'
# fi
# DPATH_ICA_RESULTS="${DPATH_OUT_ICA}/${ICA_RESULTS_PREFIX}${COHORT_ID_DBM_LIST}${ICA_SUFFIX}"
# if [[ ! -d $DPATH_ICA_RESULTS || -z "$(ls -A ${DPATH_ICA_RESULTS})" ]]
# then
	
# 	# run ICA
# 	message "ICA results not found. Computing now (this might take a while)."
# 	${FPATH_MRI_CODE} ica \
# 		${FPATH_DBM_LIST_COHORT} \
# 		${DPATH_OUT_DBM} \
# 		${DPATH_ICA_RESULTS} \
# 		${DIM_FLAG} \
# 		${SHUFFLE_FLAG} \
# 		--overwrite \
# 		--logfile ${DPATH_ICA_RESULTS}/ica.log
# 	exit_if_error $?

# else
# 	message "Found ICA results!"
# fi

# message "COPYING COHORT FILES"
# cp $FPATH_PROC_STATUS_COHORT $FPATH_DBM_LIST_COHORT $DPATH_ICA_RESULTS

# echo "FPATH_PROC_STATUS_COHORT=${FPATH_PROC_STATUS_COHORT}"
# echo "FPATH_DBM_LIST_COHORT=${FPATH_DBM_LIST_COHORT}"
# echo "DPATH_ICA_RESULTS=${DPATH_ICA_RESULTS}"
# echo "FINAL_COHORT_ID=${COHORT_ID_DBM_LIST}"

# ========================================
# delete temp file
# ========================================
rm $FPATH_TMP

# return to original directory
cd $DPATH_INITIAL
