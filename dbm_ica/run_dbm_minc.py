#!/usr/bin/env python
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Union

import click
import pandas as pd

from bids import BIDSLayout
from bids.layout import parse_file_entities

from helpers import (
    add_common_options, 
    add_dbm_minc_options,
    add_suffix, 
    callback_path, 
    check_dbm_inputs,
    ScriptHelper,
    with_helper,
)

from helpers import (
    EXT_GZIP,
    EXT_MINC,
    EXT_NIFTI,
    EXT_TRANSFORM,
    SUFFIX_T1,
    SEP_SUFFIX,
)

# output file naming for DBM pipeline
SUFFIX_DENOISED = 'denoised'
SUFFIX_NORM = 'norm_lr'
SUFFIX_MASK = '_mask' # binary mask
SUFFIX_MASKED = 'masked' # masked brain
SUFFIX_NONLINEAR = 'nlr'
SUFFIX_DBM = 'dbm'
SUFFIX_RESAMPLED = 'resampled'

EXT_LOG = '.log'
PREFIX_PIPELINE = 'dbm_minc'

# subdirectory names
DNAME_OUTPUT = 'output'
DNAME_LOGS = 'logs'

# for check_status
FNAME_STATUS = 'proc_status.csv'
STATUS_PASS = 'PASS'
STATUS_FAIL = 'FAIL'
COL_PROC_PATH = 'fpath_input'

# for multi-file command
MIN_I_FILE = 1

# job settings for multi-file command
JOB_TYPE_SLURM = 'slurm'
JOB_TYPE_SGE = 'sge'
VALID_JOB_TYPES = [JOB_TYPE_SLURM, JOB_TYPE_SGE]
DEFAULT_JOB_MEMORY = '12G'
DEFAULT_JOB_TIME = '0:20:00'
VARNAME_I_FILE = 'I_FILE'
BINDPATH_SCRIPTS = '/mnt/scripts'
BINDPATH_BIDS_DATA = '/mnt/bids'
BINDPATH_OUT = '/mnt/out'
BINDPATH_BIDS_LIST = '/mnt/bids_list'

@click.group()
def cli():
    return

@cli.command()
@click.argument('dpath_bids', type=str, callback=callback_path)
@click.argument('fpath_out', type=str, callback=callback_path)
@add_common_options()
@with_helper
def bids_generate(dpath_bids: Path, fpath_out: Path, helper: ScriptHelper):

    # make sure input directory exists
    if not dpath_bids.exists():
        helper.print_error_and_exit(f'BIDS directory not found: {dpath_bids}')

    # check if file exists
    helper.check_file(fpath_out)
    
    # create output directory
    helper.mkdir(fpath_out.parent, exist_ok=True)

    # create index for BIDS directory
    bids_layout = BIDSLayout(dpath_bids)

    # get all T1 files
    fpaths_t1 = bids_layout.get(
        extension=f'{EXT_NIFTI}{EXT_GZIP}'.strip('.'), 
        suffix=SUFFIX_T1,
        return_type='filename',
    )

    helper.echo(f'Found {len(fpaths_t1)} T1 files')

    # write paths to output file
    with fpath_out.open('w') as file_out:
        for fpath_t1 in fpaths_t1:
            fpath_t1 = Path(fpath_t1).relative_to(dpath_bids)
            file_out.write(f'{fpath_t1}\n')

@cli.command()
@click.argument('dpath_bids', callback=callback_path)
@click.argument('fpath_bids_list', callback=callback_path)
@click.argument('dpath_out', default='.', callback=callback_path)
@click.option('-i', '--i-file', 'i_file_single', 
              type=click.IntRange(min=MIN_I_FILE))
@click.option('-r', '--range', 'i_file_range', 
              type=click.IntRange(min=MIN_I_FILE), nargs=2)
@click.option('-j', '--job', 'job_type', envvar='JOB_TYPE',
              type=click.Choice(VALID_JOB_TYPES,case_sensitive=False))
@click.option('--job-resource', envvar='JOB_RESOURCE')
@click.option('--job-log', 'fpath_log_job', callback=callback_path,
              default=f'{PREFIX_PIPELINE}{EXT_LOG}', 
              envvar='FPATH_DBM_JOB_LOG')
@click.option('-c', '--container', 'fpath_container', callback=callback_path,
              envvar='FPATH_DBM_CONTAINER')
@click.option('-m', '--memory', 'job_memory', default=DEFAULT_JOB_MEMORY)
@click.option('-t', '--time', 'job_time', default=DEFAULT_JOB_TIME)
@click.option('--rename-log/--no-rename-log', default=True)
@add_dbm_minc_options()
@add_common_options()
@with_helper
def bids_run(
    dpath_bids: Path,
    fpath_bids_list: Path,
    dpath_out: Path,
    helper: ScriptHelper,
    i_file_single: Union[int, None],
    i_file_range: Union[tuple, None],
    job_type: str,
    job_resource: str,
    fpath_log_job: Path,
    fpath_container: Union[Path, None],
    job_memory: str,
    job_time: str,
    rename_log: bool,
    **kwargs,
    ):

    # make output directory now
    # need to mount it when running containing
    helper.mkdir(dpath_out, exist_ok=True)

    # convert to range
    if i_file_single is not None:
        i_file_range = (i_file_single, i_file_single)

    # get min/max of range
    if i_file_range is not None:
        i_file_min = min(i_file_range)
        i_file_max = max(i_file_range)
    else:
        i_file_min = MIN_I_FILE
        with fpath_bids_list.open() as file_bids_list:
            i_file_max = MIN_I_FILE
            for _ in file_bids_list:
                i_file_max += 1

    # submit job array
    if job_type is not None:

        # make sure job account/queue is specified
        if job_resource is None:
            helper.print_error_and_exit(
                '--job-resource must be specified when --job is given',
            )

        # make sure container is specified and exists
        if fpath_container is None:
            helper.print_error_and_exit(
                '--container must be specified when --job is given',
            )
        if not fpath_container.exists():
            helper.print_error_and_exit(
                f'Container not found: {fpath_container}'
            )

        # get path to this file
        fpath_script = Path(__file__).resolve()
        bindpath_script = Path(BINDPATH_SCRIPTS) / fpath_script.name
        dpath_scripts = fpath_script.parent

        script_command_args = [
            f'{bindpath_script} bids-run',
            f'{BINDPATH_BIDS_DATA} {BINDPATH_BIDS_LIST} {BINDPATH_OUT}',
            f'-i ${VARNAME_I_FILE}',
            '--rename-log' if rename_log else '',
            '--overwrite' if helper.overwrite else '',
        ]

        singularity_command_args = [
            'singularity', 'run',
            f'--bind {dpath_scripts}:{BINDPATH_SCRIPTS}:ro',
            f'--bind {dpath_bids}:{BINDPATH_BIDS_DATA}:ro',
            f'--bind {fpath_bids_list}:{BINDPATH_BIDS_LIST}:ro',
            f'--bind {dpath_out}:{BINDPATH_OUT}',
            f'{fpath_container}',
            ' '.join(script_command_args),
        ]
    
        # temporary file for job submission script
        with NamedTemporaryFile('w+t') as file_tmp:

            fpath_submission_tmp = Path(file_tmp.name)

            if job_type == JOB_TYPE_SGE:

                varname_array_job_id = 'SGE_TASK_ID'
                varname_job_id = 'JOB_ID'

                job_command_args = [
                    'qsub',
                    '-cwd',
                    '-N', PREFIX_PIPELINE,
                    '-q', job_resource,
                    '-t', f'{i_file_min}-{i_file_max}:1',
                    '-l', f'h_vmem={job_memory}',
                    '-l', f'h_rt={job_time}',
                    '-j', 'y',
                    '-o', fpath_log_job,
                    fpath_submission_tmp,
                ]

            else:
                raise NotImplementedError(
                    f'Not implemented for job type {job_type} yet'
                )

            # job submission script
            singularity_command = ' '.join(singularity_command_args)
            varname_command = 'COMMAND'
            submission_file_lines = [
                '#!/bin/bash',
                (
                    'echo ===================='
                    f' START JOB: ${{{varname_job_id}}} '
                    '===================='
                ),
                'echo `date`',
                f'echo "Memory: {job_memory}"',
                f'echo "Time: {job_time}"',
                f'{VARNAME_I_FILE}=${varname_array_job_id}',
                f'{varname_command}="{singularity_command}"',
                'echo "--------------------"',
                f'echo ${{{varname_command}}}',
                'echo "--------------------"',
                f'eval ${{{varname_command}}}',
                'echo `date`',
                (
                    'echo ===================='
                    f' END JOB: ${{{varname_job_id}}} '
                    '===================='
                ),
            ]

            # write to file
            for submission_file_line in submission_file_lines:
                file_tmp.write(f'{submission_file_line}\n')
            file_tmp.flush() # write right now
            fpath_submission_tmp.chmod(0o744) # make executable

            # print file
            helper.run_command(['cat', fpath_submission_tmp])

            # submit
            helper.run_command(job_command_args)
    
    # otherwise run the pipeline directly
    else:

        dpath_results = dpath_out / DNAME_OUTPUT
        dpath_logs = dpath_out / DNAME_LOGS
        helper.mkdir(dpath_results, exist_ok=True)
        layout_results = BIDSLayout(dpath_results, validate=False)

        with fpath_bids_list.open('r') as file_bids_list:

            for i_file, line in enumerate(file_bids_list, start=MIN_I_FILE):

                if i_file < i_file_min:
                    continue
                if i_file > i_file_max:
                    break

                # remove newline
                fpath_t1_relative = line.strip()

                # skip empty lines
                if fpath_t1_relative == '':
                    continue

                fpath_t1 = dpath_bids / fpath_t1_relative

                # generate path to BIDS-like output directory
                bids_entities = parse_file_entities(fpath_t1)
                dpath_out_bids = Path(layout_results.build_path(bids_entities)).parent

                fpath_log = dpath_logs / f'{PREFIX_PIPELINE}-{i_file}{EXT_LOG}'
                helper.echo(f'Running pipeline on T1 file {fpath_t1}')
                helper.echo(f'\tLog: {fpath_log}')

                _run_dbm_minc(
                    fpath_nifti=fpath_t1,
                    dpath_out=dpath_out_bids,
                    fpath_log=fpath_log, # new log file
                    rename_log=rename_log,
                    verbosity=helper.verbosity,
                    quiet=helper.quiet,
                    dry_run=helper.dry_run,
                    overwrite=helper.overwrite,
                    **kwargs,
                )

@cli.command()
@click.argument('fpath_bids_list', callback=callback_path)
@click.argument('dpath_out', callback=callback_path)
@click.option('-s', '--step', 'step_suffix_pairs', nargs=2, multiple=True, required=True)
@click.option('-o', '--file-out', 'fname_out', default=FNAME_STATUS)
@click.option('-e', '--extension-t1', 'ext_t1', default=f'{EXT_NIFTI}{EXT_GZIP}')
@add_common_options()
@with_helper
def check_status(helper: ScriptHelper, fpath_bids_list: Path, dpath_out: Path, 
                 step_suffix_pairs, fname_out, ext_t1):

    def get_fpath_t1(path_result, dpath_bids_root):
        path_result = Path(path_result)
        dpath_parent_rel = path_result.parent.relative_to(dpath_bids_root)
        prefix = path_result.name.split('.')[0]
        fname_raw = f'{prefix}{ext_t1}'
        return str(dpath_parent_rel / fname_raw)

    col_fpath_t1 = 'fpath_t1'

    dpath_bids = dpath_out / DNAME_OUTPUT
    fpath_out = (dpath_out / fname_out).resolve()

    helper.check_file(fpath_out)

    layout = BIDSLayout(dpath_bids, validate=False)
    df_layout = layout.to_df()
    df_layout[col_fpath_t1] = df_layout['path'].apply(
        lambda path: get_fpath_t1(path, dpath_bids)
    )

    if helper.verbose():
        helper.echo('\nChecking processing status for steps:')
        for step, suffix in step_suffix_pairs:
            helper.echo(f'{step}:\t{suffix}')

    t1_proc_status_all = []
    df_t1s = pd.read_csv(fpath_bids_list, header=None, names=[col_fpath_t1])
    for fpath_t1 in df_t1s[col_fpath_t1]:

        t1_proc_status = layout.parse_file_entities(fpath_t1)
        t1_proc_status[COL_PROC_PATH] = fpath_t1

        df_results = df_layout.loc[df_layout[col_fpath_t1] == fpath_t1]
        extensions = df_results['extension'].tolist()

        for step, suffix in step_suffix_pairs:
            t1_proc_status[step] = (
                STATUS_PASS
                if (suffix in extensions)
                else STATUS_FAIL
            )

        t1_proc_status_all.append(t1_proc_status)

    # make df
    df_proc_status = pd.DataFrame(t1_proc_status_all)

    # reorder columns because entities are not the same for all T1 files
    # sometimes there is an additional 'acquisition' field
    # which gets appended to the df, but we want all entities to be before
    # the other columns
    cols_proc = df_proc_status.columns.to_list()
    last_cols = [COL_PROC_PATH] + [step for step, _ in step_suffix_pairs]
    for col in last_cols:
        cols_proc.remove(col)
        cols_proc.append(col)
    df_proc_status = df_proc_status[cols_proc]

    # write file
    df_proc_status.to_csv(fpath_out, index=False, header=True)
    if helper.verbose():
        helper.echo(f'\nWrote file to {fpath_out}', text_color='blue')

@cli.command()
@click.argument('fpath_nifti', type=str, callback=callback_path)
@click.argument('dpath_out', type=str, default='.', callback=callback_path)
@add_dbm_minc_options()
@add_common_options()
def file(**kwargs):
    _run_dbm_minc(**kwargs)

@with_helper
@check_dbm_inputs
def _run_dbm_minc(helper: ScriptHelper, fpath_nifti: Path, dpath_out: Path, 
                  dpath_templates: Path, template_prefix: str, 
                  fpath_template: Path, fpath_template_mask: Path,
                  dpath_beast_lib: Path, fpath_conf: Path, 
                  save_all: bool, rename_log: bool, **kwargs):

    def apply_mask(helper: ScriptHelper, fpath_orig, fpath_mask, dpath_out=None):
        fpath_orig = Path(fpath_orig)
        if dpath_out is None:
            dpath_out = fpath_orig.parent
        dpath_out = Path(dpath_out)
        fpath_out = add_suffix(dpath_out / fpath_orig.name, SUFFIX_MASKED)
        helper.run_command([
            'minccalc',
            '-verbose',
            '-expression', 'A[0]*A[1]',
            fpath_orig,
            fpath_mask,
            fpath_out,
        ])
        return fpath_out

    def rename_log_callback(helper: ScriptHelper, fpath_new, same_parent=True):
        fpath_old = Path(helper.file_log.name)
        fpath_new = Path(fpath_new)
        if same_parent:
            fpath_new = fpath_old.parent / fpath_new
        def _rename_log_callback():
            helper.run_command(
                ['mv', '-v', fpath_old, fpath_new], 
                force=True,
            )
        return _rename_log_callback

    # make sure input file exists and has valid extension
    if not fpath_nifti.exists():
        helper.print_error_and_exit(f'Nifti file not found: {fpath_nifti}')
    valid_file_formats = (EXT_NIFTI, f'{EXT_NIFTI}{EXT_GZIP}')
    if not str(fpath_nifti).endswith(valid_file_formats):
        helper.print_error_and_exit(
            f'Invalid file format for {fpath_nifti}. '
            f'Valid extensions are: {valid_file_formats}'
        )

    with TemporaryDirectory() as dpath_tmp:

        dpath_tmp = Path(dpath_tmp)

        # if zipped file, unzip
        if fpath_nifti.suffix == EXT_GZIP:
            fpath_raw_nii = dpath_tmp / fpath_nifti.stem # drop last suffix
            with fpath_raw_nii.open('wb') as file_raw:
                helper.run_command(['zcat', fpath_nifti], stdout=file_raw)
        # else create symbolic link
        else:
            fpath_raw_nii = dpath_tmp / fpath_nifti.name # keep last suffix
            helper.run_command(['ln', '-s', fpath_nifti, fpath_raw_nii])

        # for renaming the logfile based on nifti file name
        if rename_log:
            helper.callback = rename_log_callback(
                helper, 
                f'{fpath_raw_nii.stem}{EXT_LOG}',
                same_parent=True,
            )

        # skip if output subdirectory already exists and is not empty
        helper.check_dir(dpath_out)

        # convert to minc format
        fpath_raw = dpath_tmp / fpath_raw_nii.with_suffix(EXT_MINC)
        helper.run_command(['nii2mnc', fpath_raw_nii, fpath_raw])

        # denoise
        fpath_denoised = add_suffix(fpath_raw, SUFFIX_DENOISED)
        helper.run_command(['mincnlm', '-verbose', fpath_raw, fpath_denoised])

        # normalize, scale, perform linear registration
        fpath_norm = add_suffix(fpath_denoised, SUFFIX_NORM)
        fpath_norm_transform = fpath_norm.with_suffix(EXT_TRANSFORM)
        helper.run_command([
            'beast_normalize',
            '-modeldir', dpath_templates,
            '-modelname', template_prefix,
            fpath_denoised,
            fpath_norm,
            fpath_norm_transform,
        ])

        # get brain mask
        fpath_mask = add_suffix(fpath_norm, SUFFIX_MASK, sep=SUFFIX_MASK[0])
        helper.run_command([
            'mincbeast',
            '-flip',
            '-fill',
            '-median',
            '-same_resolution',
            '-conf', fpath_conf,
            '-verbose',
            dpath_beast_lib,
            fpath_norm,
            fpath_mask,
        ])

        # extract brain
        fpath_masked = apply_mask(helper, fpath_norm, fpath_mask)

        # extract template brain
        fpath_template_masked = apply_mask(
            helper,
            fpath_template,
            fpath_template_mask,
            dpath_out=dpath_tmp,
        )

        # perform nonlinear registration
        fpath_nonlinear = add_suffix(fpath_masked, SUFFIX_NONLINEAR)
        fpath_nonlinear_transform = fpath_nonlinear.with_suffix(EXT_TRANSFORM)
        helper.run_command([
            'nlfit_s',
            '-verbose',
            '-source_mask', fpath_mask,
            '-target_mask', fpath_template_mask,
            fpath_masked,
            fpath_template_masked,
            fpath_nonlinear_transform,
            fpath_nonlinear,
        ])

        # get DBM map (not template space)
        fpath_dbm_tmp = add_suffix(fpath_nonlinear, SUFFIX_DBM)
        helper.run_command([
            'pipeline_dbm.pl',
            '-verbose',
            '--model', fpath_template,
            fpath_nonlinear_transform,
            fpath_dbm_tmp,
        ])

        # resample to template space
        fpath_dbm = add_suffix(fpath_dbm_tmp, SUFFIX_RESAMPLED)
        helper.run_command([
            'mincresample',
            '-like', fpath_template,
            fpath_dbm_tmp,
            fpath_dbm,
        ])

        # apply mask
        fpath_dbm_masked = apply_mask(helper, fpath_dbm, fpath_template_mask)

        # convert back to nifti
        fpath_dbm_nii = fpath_dbm_masked.with_suffix(EXT_NIFTI)
        helper.run_command(['mnc2nii', '-nii', fpath_dbm_masked, fpath_dbm_nii])

        # list all output files
        helper.run_command(['ls', '-lh', dpath_tmp])

        # copy all/some result files to output directory
        if save_all:
            fpaths_to_copy = dpath_tmp.iterdir()
        else:
            fpaths_to_copy = [
                fpath_denoised,     # denoised
                fpath_mask,         # brain mask (after linear registration)
                fpath_masked,       # linearly registered (masked)
                fpath_nonlinear,    # nonlinearly registered (masked)
                fpath_dbm_nii,      # DBM map
            ]

        helper.mkdir(dpath_out)
        for fpath_source in fpaths_to_copy:
            helper.run_command([
                'cp',
                '-vfp', # verbose, force overwrite, preserve metadata
                fpath_source,
                dpath_out,
            ])

        # list files in output directory
        helper.run_command(['ls', '-lh', dpath_out])

if __name__ == '__main__':
    cli()
