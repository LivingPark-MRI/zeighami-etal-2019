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
# # 	- main cohort missing 4 subjects: 2228656744226345740
# # validation cohort: _7192068301964860554
# COHORT_ID="_1606091907888136448"
# DPATH_BIDS_LIST=`dirname ${FPATH_BIDS_LIST}`
# FPATH_TMP=$(mktemp /tmp/bids_filter.XXXXXX)
# if [ ! -f ${FPATH_BIDS_LIST_ALL} ]
# then
# 	echo "File not found: ${FPATH_BIDS_LIST_ALL}"
# 	exit 1
# fi
# # ${FPATH_SCRIPT}
# FPATH_OUT=${DPATH_BIDS_LIST}/bids_list-${COHORT_ID}.txt
# COMMAND="
# 	(~/livingpark/dbm_ica/run_tmp.py bids-filter \
# 		${FPATH_BIDS_LIST_ALL} \
# 		${DPATH_ROOT}/zeighami-etal-2019-cohort-${COHORT_ID}.csv \
# 		${FPATH_OUT} \
# 		--bad-scans ${DPATH_ROOT}/bad_scans.csv \
# 		--overwrite | tee ${FPATH_TMP}) \
# 	&& NEW_COHORT_ID=\$(grep COHORT_ID ${FPATH_TMP} | awk -F '=' '{ print \$2 }' ) \
# 	&& rm ${FPATH_TMP} \
# 	&& FPATH_OUT_NEW=${DPATH_BIDS_LIST}/bids_list-\${NEW_COHORT_ID}.txt \
# 	&& mv -v ${FPATH_OUT} \${FPATH_OUT_NEW} \
# "

# # ========================================
# # submit a job to run the DBM pipeline on multiple files
# # ========================================
# COMMAND="
# 	${FPATH_SCRIPT} dbm-from-bids \
# 		${DPATH_BIDS} \
# 		${FPATH_BIDS_LIST} \
# 		${DPATH_OUT_DBM} \
# 		--job-type ${JOB_TYPE} \
# 		--job-resource ${JOB_RESOURCE} \
# 		--job-container ${FPATH_CONTAINER} \
#         --job-log-dir ${DPATH_JOB_LOGS} \
# 		-r 322 328 \
#         --overwrite \
# "

# ========================================
# check DBM processing status
# ========================================
# main cohort: _1606091907888136448
# 	- main cohort missing 4 subjects: 2228656744226345740
# validation cohort: _7192068301964860554
COHORT_ID=""
DPATH_BIDS_LIST=`dirname ${FPATH_BIDS_LIST}`
if [ ! -z $COHORT_ID ]
then
	FPATH_IN="${DPATH_BIDS_LIST}/bids_list-${COHORT_ID}.txt"
    COPY_COMMAND="&& cp -v ${DPATH_OUT_DBM}/proc_status.csv ${DPATH_OUT_DBM}/proc_status-${COHORT_ID}.csv"
else
	FPATH_IN=$FPATH_BIDS_LIST
    COPY_COMMAND=""
fi
COMMAND="
    ${FPATH_SCRIPT} dbm-status \
        ${FPATH_IN} \
        ${DPATH_OUT_DBM} \
        --step denoised .denoised.mnc \
        --step lin_reg .denoised.norm_lr.masked.mnc \
        --step lin_reg_mask .denoised.norm_lr_mask.mnc \
        --step nonlin_reg .denoised.norm_lr.masked.nlr.mnc \
        --step dbm_nii .denoised.norm_lr.masked.nlr.dbm.reshaped.masked.nii.gz \
        --overwrite \
	${COPY_COMMAND} \
"

echo ${COMMAND}
eval "${COMMAND}"
