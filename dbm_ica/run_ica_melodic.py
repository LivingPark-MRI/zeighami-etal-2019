#!/usr/bin/env python
import traceback

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from helpers import add_common_options, callback_path, process_path, ScriptHelper

DEFAULT_VERBOSITY = 2
PREFIX_MERGED = 'dbm_merged'

@click.command()
@click.argument('fpath_filenames', callback=callback_path)
@click.argument('dpath_out', default='.', callback=callback_path)
@click.option('-d', '--dim', type=int, help='Number of PCA components')
@click.option('-n', '--n-components', type=int, help='Number of ICA components')
@add_common_options()
def run_ica_melodic(fpath_filenames: Path, dpath_out: Path, dim, n_components, 
                    fpath_log: Path, overwrite, dry_run, verbosity, quiet, **kwargs):

    if fpath_log is not None:
        fpath_log.parent.mkdir(parents=True, exist_ok=True)

    with fpath_log.open('w') if (fpath_log is not None) else nullcontext() as file_log, \
        TemporaryDirectory() as dpath_tmp:

        helper = ScriptHelper(
            file_log=file_log,
            verbosity=verbosity,
            quiet=quiet,
            dry_run=dry_run,
            overwrite=overwrite,
        )

        try:

            dpath_tmp = Path(dpath_tmp)
            helper.check_nonempty(dpath_out)

            fpaths_nii_tmp = []
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
                        ['ln', '-s', fname_nii, fpath_nii_tmp],
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
