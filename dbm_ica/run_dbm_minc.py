#!/usr/bin/env python
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import click

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
)

SUFFIX_DENOISED = 'denoised'
SUFFIX_NORM = 'norm_lr'
SUFFIX_MASK = '_mask' # binary mask
SUFFIX_MASKED = 'masked' # masked brain
SUFFIX_NONLINEAR = 'nlr'
SUFFIX_DBM = 'dbm'
SUFFIX_RESAMPLED = 'resampled'

JOB_TYPE_SLURM = 'slurm'
JOB_TYPE_SGE = 'sge'
VALID_JOB_TYPES = [JOB_TYPE_SLURM, JOB_TYPE_SGE]
DEFAULT_JOB_MEMORY = '8G'
DEFAULT_JOB_TIME = '0:20:00'

@click.group()
def cli():
    return

@cli.command()
@click.argument('dpath_bids', type=str, callback=callback_path)
@click.argument('fpath_out', type=str, callback=callback_path)
@click.option('--absolute/--relative', default=False,
              help='Save absolute paths to output file.',
)
@add_common_options()
@with_helper
def bids_generate(dpath_bids: Path, fpath_out: Path, absolute: bool, helper: ScriptHelper):

    # make sure input directory exists
    if not dpath_bids.exists():
        helper.print_error_and_exit(f'BIDS directory not found: {dpath_bids}')

    # throw error if output file already exists
    if fpath_out.exists() and not helper.overwrite:
        helper.print_error_and_exit(
            f'File {fpath_out} already exists. Use --overwrite to overwrite.'
        )
    else:
        # create output directory
        fpath_out.parent.mkdir(parents=True, exist_ok=True)

    bids_layout = BIDSLayout(dpath_bids, validate=False)

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
            if not absolute:
                fpath_t1 = Path(fpath_t1).relative_to(dpath_bids)
            file_out.write(f'{fpath_t1}\n')

@cli.command()
@click.argument('fpath_bids_list', type=str, callback=callback_path)
@click.argument('dpath_out', type=str, default='.', callback=callback_path)
@click.option('-d', '--dir-bids', 'dpath_bids', callback=callback_path)
@click.option('-i', '--i-file', 'i_file_single', type=click.IntRange(min=0))
@click.option('-r', '--range', 'i_file_range', type=click.IntRange(min=0), nargs=2)
@click.option('-c', '--container', 'fpath_container', callback=callback_path)
@click.option('-j', '--job', 'job_type', type=click.Choice(VALID_JOB_TYPES, case_sensitive=False))
@click.option('--job-resource', envvar='JOB_RESOURCE')
@add_dbm_minc_options()
@add_common_options()
@with_helper
def bids_run(
    fpath_bids_list: Path,
    dpath_out: Path,
    helper: ScriptHelper,
    dpath_bids: Union[Path, None],
    i_file_single: Union[int, None],
    i_file_range: Union[tuple, None],
    fpath_container: Union[Path, None],
    job_type: str,
    job_resource: str,
    **kwargs,
    ):

    if i_file_single is not None:
        i_file_range = (i_file_single, i_file_single)

    if i_file_range is not None:
        i_file_min = min(i_file_range)
        i_file_max = max(i_file_range)
    else:
        i_file_min = -float('inf')
        i_file_max = float('inf')

    # submit job arrays if multiple input files
    if job_type is not None:

        # make sure container exists
        if not fpath_container.exists():
            raise FileNotFoundError(fpath_container)

        # get path to this file
        fpath_script = Path(__file__).parent.resolve()
    
        # TODO
        if job_type == JOB_TYPE_SGE:
            # call container
            # mount this script + input file (read-only) + output directory
            # run script in container 
            pass
        else:
            raise NotImplementedError(f'Not implemented for job type {job_type} yet')
    
    # otherwise run the pipeline directly
    else:

        # TODO add dataset_description.json (?)
        helper.check_nonempty(dpath_out)
        layout_out = BIDSLayout(dpath_out, validate=False)

        with fpath_bids_list.open('r') as file_bids_list:

            for i_file, line in enumerate(file_bids_list):

                if i_file < i_file_min:
                    continue
                if i_file > i_file_max:
                    break

                fpath_t1 = Path(line.strip())
                if not fpath_t1.exists():
                    if dpath_bids is None:
                        raise FileNotFoundError(
                            f'{fpath_t1}. Specify -d/--dir-bids if input file contains relative paths',
                        )
                    fpath_t1 = dpath_bids / fpath_t1

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

        # skip if output subdirectory already exists and is not empty
        dpath_out_sub = helper.check_nonempty(dpath_out / fpath_raw_nii.stem)

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

        for fpath_source in fpaths_to_copy:
            helper.run_command([
                'cp',
                '-vfp', # verbose, force overwrite, preserve metadata
                fpath_source,
                dpath_out_sub,
            ])

        # list files in output directory
        helper.run_command(['ls', '-lh', dpath_out_sub])

if __name__ == '__main__':
    cli()
