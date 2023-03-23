#!/bin/bash

##################################################
# INIT
# * init-env    create default .env file
##################################################

FPATH_DBM_CODE="../dbm.py" # relative to current directory

if [ $# != 1 ]
then
    echo "Usage: $0 DPATH_BIDS"
    exit 1
else
    DPATH_BIDS=$1
fi

# move to scripts directory
DPATH_INITIAL=`pwd`
cd `dirname $(realpath $0)`

# create env file
COMMAND="$FPATH_DBM_CODE init-env $DPATH_BIDS --overwrite"
echo $COMMAND
eval $COMMAND

# return to original directory
cd $DPATH_INITIAL
