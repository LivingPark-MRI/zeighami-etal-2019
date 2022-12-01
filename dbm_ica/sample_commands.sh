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
# 	${FPATH_SCRIPT} bids-list \
# 		${DPATH_BIDS} \
# 		${FPATH_BIDS_LIST} \
# 	&& \
# 	cp ${FPATH_BIDS_LIST} ${FPATH_BIDS_LIST_ALL} \
# "

# # ========================================
# # get a subset of the bids_list rows given a PPMI cohort ID and session ID
# # after this is done rename the desired file to FPATH_BIDS_LIST
# # ========================================
# # main cohort: _1606091907888136448
# # validation cohort: _7192068301964860554
# COHORT_ID="_1606091907888136448"
# SESSION_ID="1"
# DPATH_BIDS_LIST=`dirname ${FPATH_BIDS_LIST}`
# if [ ! -f ${FPATH_BIDS_LIST_ALL} ]
# then
# 	echo "File not found: ${FPATH_BIDS_LIST_ALL}"
# 	exit 1
# fi
# COMMAND="
# 	grep -f ${DPATH_ROOT}/zeighami-etal-2019-cohort-${COHORT_ID}.csv ${FPATH_BIDS_LIST_ALL} \
# 		> ${DPATH_BIDS_LIST}/bids_list-${COHORT_ID}.txt \
# 	&& \
# 	grep ses-${SESSION_ID}_ ${DPATH_BIDS_LIST}/bids_list-${COHORT_ID}.txt \
# 		> ${DPATH_BIDS_LIST}/bids_list-${COHORT_ID}-ses_${SESSION_ID}.txt \
# "

# ========================================
# submit a job to run the DBM pipeline on multiple files
# ========================================
COMMAND="
	${FPATH_SCRIPT} dbm-from-bids \
		${DPATH_BIDS} \
		${FPATH_BIDS_LIST} \
		${DPATH_OUT_DBM} \
		--job-type ${JOB_TYPE} \
		--job-resource ${JOB_RESOURCE} \
		--job-container ${FPATH_CONTAINER} \
        --job-log-dir ${DPATH_JOB_LOGS} \
		-r 1 155 \
"

# # ========================================
# # check DBM processing status
# # ========================================
# COMMAND="
#     ${FPATH_SCRIPT} dbm-status \
#         ${FPATH_BIDS_LIST} \
#         ${DPATH_OUT_DBM} \
#         --step denoised .denoised.mnc \
#         --step lin_reg .denoised.norm_lr.masked.mnc \
#         --step lin_reg_mask .denoised.norm_lr_mask.mnc \
#         --step nonlin_reg .denoised.norm_lr.masked.nlr.mnc \
#         --step dbm_nii .denoised.norm_lr.masked.nlr.dbm.resampled.masked.nii.gz \
#         --overwrite \
# "

# # ========================================
# # generate DBM result file list
# # ========================================
# COMMAND="
#     ${FPATH_SCRIPT} dbm-list \
#         ${DPATH_OUT_DBM} \
#         ${FPATH_DBM_LIST} \
#         --threshold 3 \
#         --overwrite \
# "

echo ${COMMAND}
eval "${COMMAND}"
