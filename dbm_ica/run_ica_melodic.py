#!/usr/bin/env python
import traceback

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from helpers import process_path, ScriptHelper

DEFAULT_VERBOSITY = 2
PREFIX_MERGED = 'dbm_merged'

@click.command()
@click.argument('fpath_filenames')
@click.argument('dpath_out', default='.')
@click.option('-d', '--dim', type=int, help='Number of PCA components')
@click.option('-n', '--n-components', type=int, help='Number of ICA components')
@click.option('--logfile', 'fpath_log', type=str,
              help='Path to log file')
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
def run_ica_melodic(fpath_filenames, dpath_out, dim, n_components, fpath_log, 
                    overwrite, dry_run, verbosity, quiet):

    if fpath_log is not None:
        fpath_log = process_path(fpath_log)
        fpath_log.parent.mkdir(parents=True, exist_ok=True)
    with fpath_log.open('w') if (fpath_log is not None) else nullcontext() as file_log, \
        TemporaryDirectory() as dpath_tmp:

        try:

            # override
            if quiet:
                verbosity = 0

            helper = ScriptHelper(
                file_log=file_log,
                verbosity=verbosity,
                dry_run=dry_run,
            )

            dpath_tmp = Path(dpath_tmp)
            dpath_out = process_path(dpath_out)

            try:
                dpath_out.mkdir(parents=True, exist_ok=overwrite)
                if len(list(dpath_out.iterdir())) != 0 and not overwrite:
                    raise FileExistsError
            except FileExistsError:
                helper.print_error_and_exit(
                    f'Output directory {dpath_out} is not empty. '
                    'Use --overwrite to overwrite.'
                )

            fpaths_nii_tmp = []
            fpath_filenames = process_path(fpath_filenames)
            with fpath_filenames.open('r') as file_filenames:

                for line in file_filenames:

                    line = line.strip()
                    if line == '':
                        continue

                    fname_nii = process_path(line)
                    if not fname_nii.exists():
                        helper.print_error_and_exit(f'File not found: {fname_nii}')

                    fpath_nii_tmp = dpath_tmp / fname_nii.name
                    helper.run_command(
                        [
                            'ln',
                            '-s',
                            fname_nii,
                            fpath_nii_tmp,
                        ],
                        silent=True,
                    )

                    fpaths_nii_tmp.append(fpath_nii_tmp)

            # merge into a single nifti file
            # concatenate in 4th (time) dimension
            fpath_merged = dpath_tmp / PREFIX_MERGED
            helper.run_command(['fslmerge', '-t', fpath_merged] + fpaths_nii_tmp)
            helper.run_command(['fslinfo', fpath_merged])

            if n_components is not None:
                n_components_flag = f'--numICs={n_components}'
            else:
                n_components_flag = ''

            if dim is not None:
                dim_flag = f'--dim={dim}'
            else:
                dim_flag = ''

            helper.run_command([
                'melodic',
                '-i', fpath_merged,
                '-o', dpath_out,
                dim_flag,           # number of principal components
                n_components_flag,  # number of independent components
                '--mmthresh=0',     # threshold for z-statistic map
                '--nobet',          # without brain extraction
                '--Oall',           # output everything
                '--report',         # create HTML report
                '-v',               # verbose
            ])

            helper.run_command(['ls', '-lh', dpath_out])

        except Exception:
            helper.print_error_and_exit(traceback.format_exc())

if __name__ == '__main__':
    run_ica_melodic()
