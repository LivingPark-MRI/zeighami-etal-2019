#!/bin/bash

FNAME_DOTENV=".env"

# find path to containing environment variable definitions
FPATH_CURRENT=`realpath $0`
DPATH_CURRENT=`dirname ${FPATH_CURRENT}`
FPATH_DOTENV="${DPATH_CURRENT}/${FNAME_DOTENV}"

# load environment variable
if [ ! -f "${FPATH_DOTENV}" ]
then
	echo "File not found: ${FPATH_DOTENV}"
	exit 1
fi
source ${FPATH_DOTENV}

# # ========================================
# # generate T1 file list from BIDS data directory
# # ========================================
# COMMAND="
# 	${FPATH_DBM_SCRIPT} bids-generate \
# 		${DPATH_BIDS} \
# 		${FPATH_BIDS_LIST} \
# "

# ========================================
# submit a job to run the DBM pipeline on multiple files
# ========================================
COMMAND="
	${FPATH_DBM_SCRIPT} bids-run \
		${DPATH_BIDS} \
		${FPATH_BIDS_LIST} \
		${DPATH_OUT_DBM} \
		--job-type ${JOB_TYPE} \
		--job-resource ${JOB_RESOURCE} \
		--job-container ${FPATH_DBM_CONTAINER} \
        --job-log-dir ${DPATH_JOB_LOGS} \
		-r 1 10 \
"

# # ========================================
# # check DBM processing status
# # ========================================
# COMMAND="
#     ${FPATH_DBM_SCRIPT} check-status \
#         ${FPATH_BIDS_LIST} \
#         ${DPATH_OUT_DBM} \
#         --step denoised .denoised.mnc \
#         --step lin_reg .denoised.norm_lr.masked.mnc \
#         --step lin_reg_mask .denoised.norm_lr_mask.mnc \
#         --step nonlin_reg .denoised.norm_lr.masked.nlr.mnc \
#         --step dbm_nii .denoised.norm_lr.masked.nlr.dbm.resampled.masked.nii.gz \
#         --overwrite \
# "

echo ${COMMAND}
eval "${COMMAND}"
