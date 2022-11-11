#!/usr/bin/env python
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from helpers import (
    add_common_options, 
    add_suffix, 
    callback_path, 
    ScriptHelper,
    with_helper,
)

DEFAULT_BEAST_CONF = 'default.1mm.conf'
DEFAULT_TEMPLATE = 'mni_icbm152_t1_tal_nlin_sym_09c'

ENV_VAR_DPATH_SHARE = 'MNI_DATAPATH'
DNAME_BEAST_LIB = 'beast-library-1.1'
DNAME_TEMPLATE_MAP = {
    'mni_icbm152_t1_tal_nlin_sym_09c': 'icbm152_model_09c',
    'mni_icbm152_t1_tal_nlin_sym_09a': 'icbm152_model_09a',
}
SUFFIX_TEMPLATE_MASK = '_mask' # MNI template naming convention

EXT_NIFTI = '.nii'
EXT_GZIP = '.gz'
EXT_MINC = '.mnc'
EXT_TRANSFORM = '.xfm'

SUFFIX_DENOISED = 'denoised'
SUFFIX_NORM = 'norm_lr'
SUFFIX_MASK = '_mask' # binary mask
SUFFIX_MASKED = 'masked' # masked brain
SUFFIX_NONLINEAR = 'nlr'
SUFFIX_DBM = 'dbm'
SUFFIX_RESAMPLED = 'resampled'

PREFIX_RUN = '[RUN] '
PREFIX_ERR = '[ERROR] '

# TODO wrapper function that considers BIDS things
# given a BIDS directory it computes DBM for all anatomical scans (?)
# and saves output according to BIDS standard too

@click.command()
@click.argument('fpath_nifti', type=str, callback=callback_path)
@click.argument('dpath_out', type=str, default='.', callback=callback_path)
@click.option('--share-dir', 'dpath_share', 
              callback=callback_path, envvar=ENV_VAR_DPATH_SHARE,
              help='Path to directory containing BEaST library and '
                   f'anatomical models. Uses ${ENV_VAR_DPATH_SHARE} '
                   'environment variable if not specified.')
@click.option('--template-dir', 'dpath_templates', callback=callback_path,
              help='Directory containing anatomical templates.')
@click.option('--template', 'template_prefix', default=DEFAULT_TEMPLATE,
              help='Prefix for anatomical model files. '
                   f'Valid names: {list(DNAME_TEMPLATE_MAP.keys())}. '
                   f'Default: {DEFAULT_TEMPLATE}.')
@click.option('--beast-lib-dir', 'dpath_beast_lib', callback=callback_path,
              help='Path to library directory for mincbeast.')
@click.option('--beast-conf', default=DEFAULT_BEAST_CONF,
              help='Name of configuration file for mincbeast. '
                   'Default: {DEFAULT_BEAST_CONF}.')
@click.option('--save-all/--save-subset', default=True,
              help='Save all intermediate files')
@add_common_options()
@with_helper
def run_dbm_minc(fpath_nifti: Path, dpath_out: Path, 
                 dpath_share: Path, dpath_templates: Path, template_prefix, 
                 dpath_beast_lib: Path, beast_conf, save_all, 
                 helper: ScriptHelper, **kwargs):

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

     # make sure necessary paths are given
    if dpath_share is None and (dpath_templates is None or dpath_beast_lib is None):
            helper.print_error_and_exit('If --share-dir is not given, both '
                                        '--template-dir and --beast-lib-dir '
                                        'must be specified.')
    if dpath_templates is None:
        dpath_templates = dpath_share / DNAME_TEMPLATE_MAP[template_prefix]
    if dpath_beast_lib is None:
        dpath_beast_lib = dpath_share / DNAME_BEAST_LIB

    # make sure input file exists and has valid extension
    if not fpath_nifti.exists():
        helper.print_error_and_exit(f'Nifti file not found: {fpath_nifti}')
    valid_file_formats = (EXT_NIFTI, f'{EXT_NIFTI}{EXT_GZIP}')
    if not str(fpath_nifti).endswith(valid_file_formats):
        helper.print_error_and_exit(
            f'Invalid file format for {fpath_nifti}. '
            f'Valid extensions are: {valid_file_formats}'
        )

    # generate paths for template files and make sure they are valid
    fpath_template = dpath_templates / f'{template_prefix}{EXT_MINC}'
    fpath_template_mask = add_suffix(fpath_template, 
                                    SUFFIX_TEMPLATE_MASK, sep=None)
    if not fpath_template.exists():
        helper.print_error_and_exit(f'Template file not found: {fpath_template}')
    if not fpath_template_mask.exists():
        helper.print_error_and_exit(
            f'Template mask file not found: {fpath_template_mask}'
        )

    # make sure beast library can be found
    if not dpath_beast_lib.exists():
        helper.print_error_and_exit(
            f'BEaST library directory not found: {dpath_beast_lib}'
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
        fpath_conf = dpath_beast_lib / beast_conf
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
    run_dbm_minc()
