#!/usr/bin/env python
from pathlib import Path

import click
from helpers import add_common_options, callback_path, ScriptHelper, with_helper

FNAME_DBM_LOG = 'dbm_minc.log'
FNAME_CONTAINER = 'nd-minc_1_9_16-fsl_5_0_11-click_pandas_pybids.sif'
FNAME_BIDS_LIST = 'bids_list.txt'

DNAME_OUT = 'processed'
DNAME_OUT_DBM = 'dbm'
DNAME_OUT_ICA = 'ica'

@click.command()
@click.argument('dpath-root', default='.', callback=callback_path)
@click.argument('dpath-bids', default='bids', callback=callback_path)
@click.option('-j', '--job-resource')
@click.option('-f', '--fname-dotenv', default='.env')
@add_common_options()
@with_helper
def create_default_dotenv(
    dpath_root: Path, 
    dpath_bids: Path, 
    job_resource: str,
    fname_dotenv: str, 
    helper: ScriptHelper,
    ):

    if job_resource is None:
        job_resource = ''
    
    if helper.verbose:
        helper.print_info(f'Generating default dotenv file with root directory: {dpath_root}')

    # project root directory
    constants = {
        'DPATH_ROOT': dpath_root,
        'DPATH_BIDS': dpath_bids,
        'JOB_RESOURCE': job_resource,
    }

    # MRI processing subdirectory
    constants['DPATH_MRI_SCRIPTS'] = constants['DPATH_ROOT'] / 'dbm_ica'
    constants['FPATH_DBM_SCRIPT'] = constants['DPATH_MRI_SCRIPTS'] / 'run_dbm_minc.py'
    constants['FPATH_DBM_CONTAINER'] = constants['DPATH_MRI_SCRIPTS'] / FNAME_CONTAINER
    constants['FPATH_DBM_JOB_LOG'] = constants['DPATH_MRI_SCRIPTS'] / FNAME_DBM_LOG

    # MRI output
    constants['DPATH_OUT'] = constants['DPATH_ROOT'] / DNAME_OUT
    constants['DPATH_OUT_DBM'] = constants['DPATH_OUT'] / DNAME_OUT_DBM
    constants['FPATH_BIDS_LIST'] = constants['DPATH_OUT_DBM'] / FNAME_BIDS_LIST
    constants['DPATH_OUT_ICA'] = constants['DPATH_OUT'] / DNAME_OUT_ICA

    if not Path(constants['DPATH_MRI_SCRIPTS']).exists():
        helper.print_error_and_exit(
            f'Directory not found: {constants["DPATH_MRI_SCRIPTS"]}. '
            'Make sure root directory is correct.'
        )
    
    # write dotenv file
    fpath_out = Path(constants['DPATH_MRI_SCRIPTS'], fname_dotenv)
    helper.check_file(fpath_out)
    with fpath_out.open('w') as file_dotenv:
        for key, value in constants.items():
            line = f'{key}={value}\n'
            file_dotenv.write(line)
        
    helper.print_outcome(f'Variables written to {fpath_out}', text_color='blue')

if __name__ == '__main__':
    create_default_dotenv()