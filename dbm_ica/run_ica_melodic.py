#!/usr/bin/env python
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from helpers import (
    add_common_options, 
    callback_path, 
    process_path, 
    ScriptHelper,
    with_helper, 
)

PREFIX_MERGED = 'dbm_merged'

@click.command()
@click.argument('fpath_filenames', callback=callback_path)
@click.argument('dpath_out', default='.', callback=callback_path)
@click.option('-d', '--dim', type=int, help='Number of PCA components')
@click.option('-n', '--n-components', type=int, help='Number of ICA components')
@add_common_options()
@with_helper
def run_ica_melodic(fpath_filenames: Path, dpath_out: Path, dim, n_components, 
                    helper: ScriptHelper, **kwargs):

    with TemporaryDirectory() as dpath_tmp:

        dpath_tmp = Path(dpath_tmp)
        helper.check_nonempty(dpath_out)

        # read files and make symlinks
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
        
        # check image dimensions
        helper.run_command(['fslinfo', fpath_merged])

        # melodic options
        n_components_flag = '' if (n_components is None) else f'--numICs={n_components}'
        dim_flag = '' if (dim is None) else f'--dim={dim}'

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

if __name__ == '__main__':
    run_ica_melodic()
