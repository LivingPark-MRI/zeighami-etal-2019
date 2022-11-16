#!/usr/bin/env python
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from bids import BIDSLayout

from helpers import (
    add_common_options, 
    add_dbm_minc_options,
    add_suffix, 
    callback_path, 
    check_dbm_inputs,
    EXT_GZIP,
    EXT_MINC,
    EXT_NIFTI,
    EXT_TRANSFORM,
    ScriptHelper,
    SUFFIX_T1,
    with_helper,
)

SUFFIX_DENOISED = 'denoised'
SUFFIX_NORM = 'norm_lr'
SUFFIX_MASK = '_mask' # binary mask
SUFFIX_MASKED = 'masked' # masked brain
SUFFIX_NONLINEAR = 'nlr'
SUFFIX_DBM = 'dbm'
SUFFIX_RESAMPLED = 'resampled'

@click.group()
def cli():
    return

@cli.command()
@click.argument('dpath_bids', type=str, callback=callback_path)
@click.argument('dpath_out', type=str, default='.', callback=callback_path)
@add_dbm_minc_options()
@add_common_options()
@with_helper
def bids(dpath_bids: Path, dpath_out: Path, helper: ScriptHelper, **kwargs):
    
    # make sure input directory exists
    if not dpath_bids.exists():
        helper.print_error_and_exit(f'BIDS directory not found: {dpath_bids}')

    # create output directory if it doesn't exist yet
    dpath_out.mkdir(parents=True, exist_ok=True)

    bids_layout = BIDSLayout(dpath_bids)

    fpaths_t1 = bids_layout.get(
        extension=f'{EXT_NIFTI}{EXT_GZIP}'.strip('.'), 
        suffix=SUFFIX_T1, 
        return_type='filename',
    )

    helper.echo(len(fpaths_t1))

    for fpath_t1 in fpaths_t1:
        run_dbm_minc_single_file(
            fpath_nifti=fpath_t1,
            dpath_out=dpath_out,
            helper=helper,
            **kwargs,
        )

@cli.command()
@click.argument('fpath_nifti', type=str, callback=callback_path)
@click.argument('dpath_out', type=str, default='.', callback=callback_path)
@add_dbm_minc_options()
@add_common_options()
@with_helper
@check_dbm_inputs
def file(**kwargs):
    run_dbm_minc_single_file(**kwargs)

def run_dbm_minc_single_file(fpath_nifti: Path, dpath_out: Path, 
                             dpath_templates: Path, template_prefix: str, 
                             fpath_template: Path, fpath_template_mask: Path,
                             dpath_beast_lib: Path, fpath_conf: Path, 
                             save_all, helper: ScriptHelper):

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

    helper.timestamp()

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

        helper.timestamp()

if __name__ == '__main__':
    cli()
