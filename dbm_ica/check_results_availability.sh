#!/bin/bash

# settings
DPATH_BIDS_DEFAULT="/data/pd/ppmi/releases/PPMI-ver_T1/bids"
FNAME_DOTENV=".env"
FNAME_DOTENV_SCRIPT="create_default_dotenv.py"
COHORT_PREFIX="zeighami-etal-2019-cohort-"
FNAME_BAD_SCANS="bad_scans.csv"
ICA_RESULTS_PREFIX="output-"

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
if [[ $# -lt 1 || $# -gt 2 ]]
then
	echo "Usage: $0 COHORT_ID [DPATH_BIDS]"
	exit 1
else
	COHORT_ID=$1

	if [ ! -z $2 ]
	then
		DPATH_BIDS=$2
	else
		DPATH_BIDS=$DPATH_BIDS_DEFAULT
	fi
fi

message "COHORT_ID: $COHORT_ID"

# ========================================
# load environment variables
# ========================================

# find path to containing environment variable definitions
FPATH_CURRENT=`realpath $0`
DPATH_CURRENT=`dirname ${FPATH_CURRENT}`
FPATH_DOTENV="${DPATH_CURRENT}/${FNAME_DOTENV}"

# if env file does not exist, create it
if [ ! -f $FPATH_DOTENV ]
then
	echo "Did not find dotenv file. Generating default one..."
	echo "DPATH_BIDS: $DPATH_BIDS"
	$DPATH_CURRENT/$FNAME_DOTENV_SCRIPT \
		.. \
		$FNAME_DOTENV_SCRIPT \
		--fname-dotenv $FPATH_DOTENV

	if [ ! -f $FPATH_DOTENV ]
	then
		echo "ERROR when creating default dotenv file"
		exit 2
	fi
fi
source ${FPATH_DOTENV}

# ========================================
# make sure cohort file exists
# ========================================
FPATH_COHORT="${DPATH_ROOT}/${COHORT_PREFIX}${COHORT_ID}.csv" # TODO use arg for this
if [ ! -f $FPATH_COHORT ]
then
	echo "Cohort file not found: $FPATH_COHORT"
	exit 3
fi

# ========================================
# make sure BIDS list file exists
# TODO test this
# ========================================
if [ ! -f ${FPATH_BIDS_LIST_ALL} ]
then
	echo "Did not find ${FPATH_BIDS_LIST_ALL}. Generating new one..."
	
	${FPATH_SCRIPT} bids-list \
		${DPATH_BIDS} \
		${FPATH_BIDS_LIST_ALL}

	exit_if_error $?
fi

# ========================================
# filter list of T1 filepaths based on cohort
# ========================================
FPATH_BAD_SCANS="${DPATH_ROOT}/${FNAME_BAD_SCANS}"
DPATH_BIDS_LIST=`dirname ${FPATH_BIDS_LIST_ALL}`
FPATH_BIDS_LIST_COHORT=${DPATH_BIDS_LIST}/bids_list-${COHORT_ID}.txt

if [ ! -f $FPATH_BAD_SCANS ]
then
	echo "Did not find bad scans file. Creating empty one..."
	touch $FPATH_BAD_SCANS
fi

message "BIDS-FILTER"
${FPATH_SCRIPT} bids-filter \
	${FPATH_BIDS_LIST_ALL} \
	${FPATH_COHORT} \
	${FPATH_BIDS_LIST_COHORT} \
	--bad-scans ${FPATH_BAD_SCANS} \
	--overwrite | tee ${FPATH_TMP}
exit_if_error $?

COHORT_ID_BIDS_LIST=$(grep COHORT_ID ${FPATH_TMP} | awk -F '=' '{ print $2 }' )

if [ $COHORT_ID_BIDS_LIST != $COHORT_ID ]
then
	message "COHORT ID CHANGED TO ${COHORT_ID_BIDS_LIST} AFTER GENERATING BIDS LIST"
	FPATH_BIDS_LIST_COHORT_TMP=${FPATH_BIDS_LIST_COHORT}
	FPATH_BIDS_LIST_COHORT=${FPATH_BIDS_LIST_COHORT/"$COHORT_ID"/"$COHORT_ID_BIDS_LIST"}
	mv -v ${FPATH_BIDS_LIST_COHORT_TMP} ${FPATH_BIDS_LIST_COHORT}
	exit_if_error $?
fi

# ========================================
# check DBM processing status
# ========================================
message "DBM-STATUS"
${FPATH_SCRIPT} dbm-status \
	${FPATH_BIDS_LIST_COHORT} \
	${DPATH_OUT_DBM} \
	--step denoised .denoised.mnc \
	--step lin_reg .denoised.norm_lr.masked.mnc \
	--step lin_reg_mask .denoised.norm_lr_mask.mnc \
	--step nonlin_reg .denoised.norm_lr.masked.nlr.mnc \
	--step dbm_nii .denoised.norm_lr.masked.nlr.dbm.reshaped.masked.nii.gz \
	--overwrite
exit_if_error $?

message "COPYING"
FPATH_PROC_STATUS_COHORT="${DPATH_OUT_DBM}/proc_status-${COHORT_ID}.csv"
cp -v ${DPATH_OUT_DBM}/proc_status.csv ${FPATH_PROC_STATUS_COHORT}
exit_if_error $?

# ========================================
# build list of filenames for ICA
# ========================================
message "DBM-LIST"
${FPATH_SCRIPT} dbm-list \
	${DPATH_OUT_DBM} \
	${FPATH_DBM_LIST} \
	--overwrite | tee ${FPATH_TMP}
exit_if_error $?

COHORT_ID_DBM_LIST=$(grep COHORT_ID ${FPATH_TMP} | awk -F '=' '{ print $2 }' ) \

if [ $COHORT_ID_DBM_LIST != $COHORT_ID_BIDS_LIST ]
then
	message "COHORT ID CHANGED TO ${COHORT_ID_DBM_LIST} AFTER GENERATING DBM LIST"
fi

message "COPYING"
FPATH_DBM_LIST_COHORT=${DPATH_OUT_ICA}/dbm_list-${COHORT_ID_DBM_LIST}.txt
cp -v ${FPATH_DBM_LIST} ${FPATH_DBM_LIST_COHORT}
exit_if_error $?

# ========================================
# check if ICA results exist
# ========================================
DIM="30"
# DIMEST="aic" # 'lap', 'bic', 'mdl', 'aic', 'mean'
# SHUFFLE="y" # shuffles if non-empty
SEP_SUFFIX="-"
if [[ ! (-z $DIM || -z $DIMEST) ]]
then
	message "Only one of DIM and DIMEST can be set (got $DIM and $DIMEST)"
elif [ ! -z $DIM ]
then
	DIM_FLAG="--dim $DIM"
	ICA_SUFFIX="${SEP_SUFFIX}${DIM}"
elif [ ! -z $DIMEST ]
then
	DIM_FLAG="--dimest $DIMEST"
	ICA_SUFFIX="${SEP_SUFFIX}${DIMEST}"
fi
if [ ! -z $SHUFFLE ]
then
	ICA_SUFFIX="${ICA_SUFFIX}_shuffle"
	SHUFFLE_FLAG='--shuffle'
fi
DPATH_ICA_RESULTS="${DPATH_OUT_ICA}/${ICA_RESULTS_PREFIX}${COHORT_ID_DBM_LIST}${ICA_SUFFIX}"
if [[ ! -d $DPATH_ICA_RESULTS || -z "$(ls -A ${DPATH_ICA_RESULTS})" ]]
then
	
	# run ICA
	message "ICA results not found. Computing now (this might take a while)."
	${FPATH_SCRIPT} ica \
		${FPATH_DBM_LIST_COHORT} \
		${DPATH_OUT_DBM} \
		${DPATH_ICA_RESULTS} \
		${DIM_FLAG} \
		${SHUFFLE_FLAG} \
		--overwrite \
		--logfile ${DPATH_ICA_RESULTS}/ica.log
	exit_if_error $?

else
	message "Found ICA results!"
fi

message "COPYING COHORT FILES"
cp $FPATH_PROC_STATUS_COHORT $FPATH_DBM_LIST_COHORT $DPATH_ICA_RESULTS

echo "FPATH_PROC_STATUS_COHORT=${FPATH_PROC_STATUS_COHORT}"
echo "FPATH_DBM_LIST_COHORT=${FPATH_DBM_LIST_COHORT}"
echo "DPATH_ICA_RESULTS=${DPATH_ICA_RESULTS}"
echo "FINAL_COHORT_ID=${COHORT_ID_DBM_LIST}"

# ========================================
# delete temp file
# ========================================
rm $FPATH_TMP
