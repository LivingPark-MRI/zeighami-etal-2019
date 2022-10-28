#!/usr/bin/env python
import sys
import subprocess
import shutil
import traceback
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import Iterable, Union

import click

DEFAULT_VERBOSITY = 2
DEFAULT_BEAST_CONF = 'default.1mm.conf'
DEFAULT_TEMPLATE = 'mni_icbm152_t1_tal_nlin_sym_09c'

ENV_VAR_DPATH_SHARE = 'MNI_DATAPATH'
DNAME_BEAST_LIB = 'beast-library-1.1'
DNAME_TEMPLATE_MAP = {
    'mni_icbm152_t1_tal_nlin_sym_09c': 'icbm152_model_09c',
    'mni_icbm152_t1_tal_nlin_sym_09a': 'icbm152_model_09a',
}
SUFFIX_TEMPLATE_MASK = '_mask'

EXT_NIFTI = '.nii'
EXT_GZIP = '.gz'
EXT_MINC = '.mnc'
EXT_TRANSFORM = '.xfm'

SUFFIX_DENOISED = 'denoised'
SUFFIX_NORM = 'norm'
SUFFIX_MASK = 'mask'
SUFFIX_EXTRACTED = 'extracted'
SUFFIX_NONLINEAR = 'nl'
SUFFIX_DBM_TMP = 'dbm'
SUFFIX_DBM = 'sub1'

PREFIX_RUN = '[RUN] '
PREFIX_ERR = '[ERROR] '

# TODO wrapper function that considers BIDS things
# given a BIDS directory it computes DBM for all anatomical scans (?)
# and saves output according to BIDS standdard too (what would that be like?)
# TODO print start/end time
# TODO write stdout/stderr to log file (from wrapper function?)

@click.command()
@click.argument('fpath_nifti', type=str)
@click.argument('dpath_out', type=str, default='.')
@click.option('--share-dir', 'dpath_share', envvar=ENV_VAR_DPATH_SHARE,
              help='Path to directory containing BEaST library and '
                   f'anatomical models. Uses ${ENV_VAR_DPATH_SHARE} '
                   'environment variable if not specified.')
@click.option('--template-dir', 'dpath_templates',
              help='Directory containing anatomical templates.')
@click.option('--template', 'template_prefix', default=DEFAULT_TEMPLATE,
              help='Prefix for anatomical model files. '
                   f'Valid names: {list(DNAME_TEMPLATE_MAP.keys())}. '
                   f'Default: {DEFAULT_TEMPLATE}.')
@click.option('--beast-lib-dir', 'dpath_beast_lib', 
              help='Path to library directory for mincbeast.')
@click.option('--beast-conf', default=DEFAULT_BEAST_CONF,
              help='Name of configuration file for mincbeast. '
                   'Default: {DEFAULT_BEAST_CONF}.')
@click.option('--overwrite/--no-overwrite', default=False,
              help='Overwrite existing result files.')
@click.option('--dry-run/--no-dry-run', default=False,
              help='Print shell commands without executing them.')
@click.option('-v', '--verbose', 'verbosity', count=True, 
              default=DEFAULT_VERBOSITY,
              help='Set/increase verbosity level (cumulative). '
                   f'Default level: {DEFAULT_VERBOSITY}.')
@click.option('--quiet', is_flag=True, default=False,
              help='Suppress output whenever possible. '
                   'Has priority over -v/--verbose flags.')
def run_dbm_minc(fpath_nifti, dpath_out, dpath_share, 
                 dpath_templates, template_prefix, dpath_beast_lib, beast_conf, 
                 overwrite, dry_run, verbosity, quiet):

    def run_command(args, shell=False, stdout=None, stderr=None):
        args = [str(arg) for arg in args if arg != '']
        args_str = ' '.join(args)
        if (verbosity > 0) or dry_run:
            echo(f'{args_str}', prefix=PREFIX_RUN, text_color='yellow',
                 color_prefix_only=dry_run)
        if not dry_run:
            try:
                subprocess.run(args, check=True, shell=shell,
                               stdout=stdout, stderr=stderr)
            except subprocess.CalledProcessError as ex:
                print_error_and_exit(
                    f'Command {args_str} returned {ex.returncode}',
                    exit_code=ex.returncode,
                )

    try:

        # overwrite if needed
        if quiet:
            verbosity = 0

        # to pass to MINC tools
        if verbosity > 1:
            minc_verbose_flag = '-verbose'
            minc_quiet_flag = ''
        else:
            minc_verbose_flag = ''
            minc_quiet_flag = '-quiet'

        # process paths
        fpath_nifti = Path(fpath_nifti).expanduser().absolute()
        dpath_out = Path(dpath_out).expanduser().absolute()
        if dpath_share is None:
            if (dpath_templates is None) or (dpath_beast_lib is None):
                print_error_and_exit('If --share-dir is not given, '
                                    'both --template-dir and --beast-lib-dir '
                                    'must be specified.')
        else:
            dpath_share = Path(dpath_share).expanduser().absolute()
        if dpath_templates is None:
            dpath_templates = dpath_share / DNAME_TEMPLATE_MAP[template_prefix]
        else:
            dpath_templates = Path(dpath_templates).expanduser().absolute()
        if dpath_beast_lib is None:
            dpath_beast_lib = dpath_share / DNAME_BEAST_LIB
        else:
            dpath_beast_lib = Path(dpath_beast_lib).expanduser().absolute()

        # make sure input file exists and has valid extension
        if not fpath_nifti.exists():
            print_error_and_exit(f'Nifti file not found: {fpath_nifti}')
        valid_file_formats = (EXT_NIFTI, f'{EXT_NIFTI}{EXT_GZIP}')
        if not str(fpath_nifti).endswith(valid_file_formats):
            print_error_and_exit(
                f'Invalid file format for {fpath_nifti}'
                f'. Valid extensions are: {valid_file_formats}'
            )

        # generate paths for template files and make sure they are valid
        fpath_template = dpath_templates / f'{template_prefix}{EXT_MINC}'
        fpath_template_mask = add_suffix(fpath_template, 
                                         SUFFIX_TEMPLATE_MASK, sep=None)
        if not fpath_template.exists():
            print_error_and_exit(f'Template file not found: {fpath_template}')
        if not fpath_template_mask.exists():
            print_error_and_exit(
                f'Template mask file not found: {fpath_template_mask}'
            )

        # make sure beast library can be found
        if not dpath_beast_lib.exists():
            print_error_and_exit(
                f'BEaST library directory not found: {dpath_beast_lib}'
            )

        with TemporaryDirectory() as dpath_tmp:
            dpath_tmp = Path(dpath_tmp)

            # if zipped file, unzip
            if fpath_nifti.suffix == EXT_GZIP:
                fpath_raw_nii = dpath_tmp / fpath_nifti.stem # drop last suffix
                with fpath_raw_nii.open('wb') as file_raw:
                    run_command(['zcat', fpath_nifti], stdout=file_raw)
            # else create symbolic link
            else:
                fpath_raw_nii = dpath_tmp / fpath_nifti.name # keep last suffix
                run_command(['ln', '-s', fpath_nifti, fpath_raw_nii])

            # skip if output subdirectory already exists
            dpath_out_sub = dpath_out / fpath_raw_nii.stem
            try:
                dpath_out_sub.mkdir(parents=True, exist_ok=overwrite)
            except FileExistsError:
                if len(list(dpath_out_sub.iterdir())) != 0:
                    print_error_and_exit(
                        f'Non-empty output directory {dpath_out_sub} '
                        'already exists. Use --overwrite to overwrite.'
                    )

            # convert to minc format
            fpath_raw = dpath_tmp / fpath_raw_nii.with_suffix(EXT_MINC)
            run_command([
                'nii2mnc', 
                minc_quiet_flag, 
                fpath_raw_nii, 
                fpath_raw,
            ])

            # denoise
            fpath_denoised = add_suffix(fpath_raw, SUFFIX_DENOISED)
            run_command([
                'mincnlm', 
                minc_verbose_flag,
                fpath_raw, 
                fpath_denoised,
            ])

            # normalize, scale, perform linear registration
            fpath_norm = add_suffix(fpath_denoised, SUFFIX_NORM)
            fpath_norm_transform = fpath_norm.with_suffix(EXT_TRANSFORM)
            run_command([
                'beast_normalize', 
                '-modeldir', dpath_templates,
                '-modelname', template_prefix,
                fpath_denoised, 
                fpath_norm, 
                fpath_norm_transform,
            ])

            # get brain mask
            fpath_mask = add_suffix(fpath_norm, SUFFIX_MASK)
            fpath_conf = dpath_beast_lib / beast_conf
            run_command([
                'mincbeast',
                '-fill',
                '-median',
                '-conf', fpath_conf,
                minc_verbose_flag,
                dpath_beast_lib,
                fpath_norm,
                fpath_mask,
            ])

            # extract brain
            fpath_extracted = add_suffix(fpath_norm, SUFFIX_EXTRACTED)
            run_command([
                'minccalc',
                minc_verbose_flag,
                '-expression', 'A[0]*A[1]',
                fpath_norm,
                fpath_mask,
                fpath_extracted,
            ])

            # perform nonlinear registration
            fpath_nonlinear = add_suffix(fpath_extracted, SUFFIX_NONLINEAR)
            fpath_nonlinear_transform = fpath_nonlinear.with_suffix(EXT_TRANSFORM)
            run_command([
                'nlfit_s',
                minc_verbose_flag,
                '-target_mask', fpath_template_mask,
                fpath_extracted,
                fpath_template,
                fpath_nonlinear_transform,
                fpath_nonlinear,
            ])

            # get DBM map
            fpath_dbm_tmp = add_suffix(fpath_nonlinear, SUFFIX_DBM_TMP)
            run_command([
                'pipeline_dbm.pl',
                minc_verbose_flag,
                '--model', fpath_template,
                fpath_nonlinear_transform,
                fpath_dbm_tmp,
            ])

            # subtract 1
            fpath_dbm = add_suffix(fpath_dbm_tmp, SUFFIX_DBM)
            run_command([
                'mincmath',
                '-sub',
                '-constant', 1,
                fpath_dbm_tmp,
                fpath_dbm,
            ])

            # convert back to nifti
            fpath_dbm_nii = fpath_dbm.with_suffix(EXT_NIFTI)
            run_command([
                'mnc2nii',
                '-nii',
                fpath_dbm,
                fpath_dbm_nii,
            ])

            # list all output files
            run_command(['ls', '-lh', dpath_tmp])

            # copy some result files to output directory
            if not dry_run:
                copy_to_dir(
                    [
                        fpath_denoised,     # denoised
                        fpath_mask,         # brain mask
                        fpath_extracted,    # linearly registered
                        fpath_nonlinear,    # nonlinearly registered
                        fpath_dbm_nii,      # DBM map
                    ], 
                    dpath_out_sub,
                    overwrite=overwrite,
                )

            # list files in output directory
            run_command(['ls', '-lh', dpath_out_sub])

    except Exception:
        print_error_and_exit(traceback.format_exc())

def echo(message, prefix='', text_color=None, color_prefix_only=False):
    if (prefix != '') and (color_prefix_only):
        text = f'{click.style(prefix, fg=text_color)}{message}'
    else:
        text = click.style(f'{prefix}{message}', fg=text_color)
    click.echo(text, color=True)

def print_error_and_exit(message, prefix=PREFIX_ERR, text_color='red', exit_code=1):
    echo(f'{prefix}{message}', text_color=text_color)
    sys.exit(exit_code)

def add_suffix(
    path: Union[Path, str], 
    suffix: str, 
    sep: Union[str, None] = '.',
) -> Path:
    if sep is not None:
        if suffix.startswith(sep):
            suffix = suffix[len(sep):]
    else:
        sep = ''
    path = Path(path)
    return path.parent / f'{path.stem}{sep}{suffix}{path.suffix}'

def copy_to_dir(
    path_source: Union[Path, str, Iterable[Union[Path, str]]], 
    dpath_target: Union[Path, str],
    overwrite=False,
):

    if isinstance(path_source, Iterable):
        for individual_path in path_source:
            copy_to_dir(individual_path, dpath_target, overwrite=overwrite)

    elif isinstance(path_source, str) or isinstance(path_source, Path):
        path_source = Path(path_source)
        dpath_target = Path(dpath_target)
        
        path_target = dpath_target / path_source.name
        if path_target.exists() and not overwrite:
            raise FileExistsError(f'{path_target} already exists.')

        shutil.copy2(path_source, path_target)

    else:
        raise ValueError(f'Invalid input: {path_source}')

if __name__ == '__main__':
    run_dbm_minc()
