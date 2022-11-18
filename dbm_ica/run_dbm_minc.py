#!/usr/bin/env python
from pathlib import Path
from tempfile import TemporaryDirectory, NamedTemporaryFile
from typing import Union

import click

from bids import BIDSLayout
from bids.layout import parse_file_entities

# TODO write to proc_status.csv
# with columns ID/session/modality/suffix
# and pass/fail for each step (conversion, denoise, linear, beast, nonlinear, DBM, etc.)
# use pandas?

# TODO move logs directory outside of output directory

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
)

SUFFIX_DENOISED = 'denoised'
SUFFIX_NORM = 'norm_lr'
SUFFIX_MASK = '_mask' # binary mask
SUFFIX_MASKED = 'masked' # masked brain
SUFFIX_NONLINEAR = 'nlr'
SUFFIX_DBM = 'dbm'
SUFFIX_RESAMPLED = 'resampled'

MIN_I_FILE = 1
DNAME_LOGS = 'logs'

# job settings
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
@click.option('-i', '--i-file', 'i_file_single', type=click.IntRange(min=MIN_I_FILE))
@click.option('-r', '--range', 'i_file_range', type=click.IntRange(min=MIN_I_FILE), nargs=2)
@click.option('-c', '--container', 'fpath_container', callback=callback_path)
@click.option('-j', '--job', 'job_type', type=click.Choice(VALID_JOB_TYPES, case_sensitive=False))
@click.option('-m', '--memory', 'job_memory', default=DEFAULT_JOB_MEMORY)
@click.option('-t', '--time', 'job_time', default=DEFAULT_JOB_TIME)
@click.option('--job-resource', envvar='JOB_RESOURCE')
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
    fpath_container: Union[Path, None],
    job_type: str,
    job_resource: str,
    job_memory: str,
    job_time: str,
    **kwargs,
    ):

    helper.mkdir(dpath_out, exist_ok=True)

    if i_file_single is not None:
        i_file_range = (i_file_single, i_file_single)

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
                '--job-resource must be specified when --job is given.',
            )

        # make sure container is specified and exists
        if fpath_container is None:
            helper.print_error_and_exit('--container must be specified when --job is given')
        if not fpath_container.exists():
            helper.print_error_and_exit(f'Container not found: {fpath_container}')

        # get path to this file
        fpath_script = Path(__file__).resolve()
        bindpath_script = Path(BINDPATH_SCRIPTS) / fpath_script.name
        dpath_scripts = fpath_script.parent

        script_command = [
            f'{bindpath_script} bids-run',
            f'{BINDPATH_BIDS_DATA} {BINDPATH_BIDS_LIST} {BINDPATH_OUT}',
            f'-i ${VARNAME_I_FILE}',
            f'--logfile {BINDPATH_OUT}/{DNAME_LOGS}/dbm_minc-${{{VARNAME_I_FILE}}}.log',
            '--rename-log',
            '--overwrite' if helper.overwrite else '',
        ]

        singularity_command = [
            'singularity', 'run',
            f'--bind {dpath_scripts}:{BINDPATH_SCRIPTS}:ro',
            f'--bind {dpath_bids}:{BINDPATH_BIDS_DATA}:ro',
            f'--bind {fpath_bids_list}:{BINDPATH_BIDS_LIST}:ro',
            f'--bind {dpath_out}:{BINDPATH_OUT}',
            f'--bind /data/origami/livingpark/zeighami-etal-2019/dbm_ica/ignore/tmp_home:/home/bic/mwang', # TODO remove
            f'{fpath_container}',
            ' '.join(script_command),
        ]
    
        # temporary file for job submission script
        with NamedTemporaryFile('w+t') as file_tmp:

            fpath_submission_tmp = Path(file_tmp.name)

            if job_type == JOB_TYPE_SGE:

                varname_array_job_id = 'SGE_TASK_ID'

                job_command_args = [
                    'qsub',
                    '-q', job_resource,
                    '-t', f'{i_file_min}-{i_file_max}:1',
                    '-l', f'h_vmem={job_memory}',
                    '-l', f'h_rt={job_time}',
                    fpath_submission_tmp,
                ]

            else:
                raise NotImplementedError(f'Not implemented for job type {job_type} yet')

            # job submission script
            submission_file_lines = [
                '#!/bin/bash',
                f'{VARNAME_I_FILE}=${varname_array_job_id}',
                ' '.join(singularity_command),
            ]

            # write to file
            for submission_file_line in submission_file_lines:
                file_tmp.write(f'{submission_file_line}\n')
            file_tmp.flush()
            fpath_submission_tmp.chmod(0o744)

            # print file
            helper.run_command(['cat', fpath_submission_tmp])

            # submit
            helper.run_command(job_command_args)
    
    # otherwise run the pipeline directly
    else:

        layout_out = BIDSLayout(dpath_out, validate=False)

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
                dpath_out_bids = Path(layout_out.build_path(bids_entities)).parent

                _run_dbm_minc(
                    fpath_nifti=fpath_t1,
                    dpath_out=dpath_out_bids,
                    helper=helper,
                    **kwargs,
                )

@cli.command()
@click.argument('fpath_nifti', type=str, callback=callback_path)
@click.argument('dpath_out', type=str, default='.', callback=callback_path)
@add_dbm_minc_options()
@add_common_options()
@with_helper
def file(**kwargs):
    _run_dbm_minc(**kwargs)

@check_dbm_inputs
def _run_dbm_minc(helper: ScriptHelper, fpath_nifti: Path, dpath_out: Path, 
                  dpath_templates: Path, template_prefix: str, 
                  fpath_template: Path, fpath_template_mask: Path,
                  dpath_beast_lib: Path, fpath_conf: Path, 
                  save_all, **kwargs):

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

        # for renaming the logfile, if needed
        helper.nifti_prefix = fpath_raw_nii.stem

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
