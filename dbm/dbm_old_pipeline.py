#!/usr/bin/env python

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

import click
import pandas as pd

from helpers import (
    add_dbm_minc_options,
    add_helper_options,
    add_suffix,
    callback_path,
    check_dbm_inputs,
    EXT_GZIP,
    EXT_JPEG,
    EXT_MINC,
    EXT_NIFTI,
    EXT_TRANSFORM,
    minc_qc,
    SEP_SUFFIX,
    ScriptHelper,
    with_helper,
)

from tracker import check_files, KW_PIPELINE_COMPLETE, KW_PHASE

# parameters
DEFAULT_COMPRESS_NII = True
FNAME_CONTAINER = 'nd-minc_1_9_16-fsl_5_0_11-click_livingpark_pandas_pybids.sif' # TODO remove FSL (?)
JOB_MEMORY = "16G"
N_THREADS_JOBS = 8
PREFIX_PIPELINE = "dbm-"

# output file naming for DBM pipeline
SUFFIX_DENOISED = "denoised"
SUFFIX_NORM = "norm_lr"
SUFFIX_MASK = "_mask"       # binary mask
SUFFIX_MASKED = "masked"    # masked brain
SUFFIX_NONLINEAR = "nlr"
SUFFIX_DBM = "dbm"
SUFFIX_RESHAPED = 'reshaped'
SUFFIX_RESAMPLED = "resampled"
SUFFIX_GRID = "_grid_0"     # MINC convention
EXT_LOG = ".log"
NLR_LEVEL_PARAM = '_level'
DBM_FWHM_PARAM = '_fwhm'

QC_FILE_PATTERNS_OLD_PIPELINE = {
    "linear": "qc_{}_{}-norm_lr.jpg", # subject session
    "linear_mask": "qc_{}_{}-norm_lr_mask.jpg",
    "nonlinear": "qc_{}_{}-nlr.jpg",
}

def check_results_old_pipeline(subject_dir, session_id, result_filenames: Iterable[str|Path]):
    results_dir = Path(subject_dir, session_id)
    return check_files(results_dir=results_dir, result_filenames=result_filenames)


def get_components_old_pipeline(prefix, nlr_level, dbm_fwhm, i_step):
    components = [
        prefix, 
        SUFFIX_DENOISED, 
        SUFFIX_NORM, 
        SUFFIX_MASKED, 
        f'{SUFFIX_NONLINEAR}{NLR_LEVEL_PARAM}{int(nlr_level)}',
        SEP_SUFFIX.join([f'{SUFFIX_DBM}{DBM_FWHM_PARAM}{int(dbm_fwhm)}', SUFFIX_RESHAPED, SUFFIX_MASKED]),
    ]
    return components[:i_step+2]


def check_preprocessing_old_pipeline(subject_dir, session_id, prefix, nlr_level, dbm_fwhm, **kwargs):
    components = get_components_old_pipeline(prefix, nlr_level=nlr_level, dbm_fwhm=dbm_fwhm, i_step=0)
    filename_denoised = Path(SEP_SUFFIX.join(components)).with_suffix(EXT_MINC)
    return check_results_old_pipeline(
        subject_dir=subject_dir,
        session_id=session_id,
        result_filenames=[filename_denoised],
    )


def check_linear_registration_old_pipeline(subject_dir, session_id, prefix, nlr_level, dbm_fwhm, **kwargs):
    components = get_components_old_pipeline(prefix, nlr_level=nlr_level, dbm_fwhm=dbm_fwhm, i_step=1)
    filename_linear = Path(SEP_SUFFIX.join(components)).with_suffix(EXT_MINC)
    return check_results_old_pipeline(
        subject_dir=subject_dir,
        session_id=session_id,
        result_filenames=[
            filename_linear, 
            filename_linear.with_suffix(EXT_TRANSFORM),
        ],        
    )


def check_brain_extraction_old_pipeline(subject_dir, session_id, prefix, nlr_level, dbm_fwhm, **kwargs):
    components = get_components_old_pipeline(prefix, nlr_level=nlr_level, dbm_fwhm=dbm_fwhm, i_step=2)
    filename_linear_masked = Path(SEP_SUFFIX.join(components)).with_suffix(EXT_MINC)
    return check_results_old_pipeline(
        subject_dir=subject_dir,
        session_id=session_id,
        result_filenames=[filename_linear_masked],        
    )


def check_nonlinear_registration_old_pipeline(subject_dir, session_id, prefix, nlr_level, dbm_fwhm, **kwargs):
    components = get_components_old_pipeline(prefix, nlr_level=nlr_level, dbm_fwhm=dbm_fwhm, i_step=3)
    filename_nonlinear = Path(SEP_SUFFIX.join(components)).with_suffix(EXT_MINC)
    return check_results_old_pipeline(
        subject_dir=subject_dir,
        session_id=session_id,
        result_filenames=[filename_nonlinear],
    )

def check_dbm_old_pipeline(subject_dir, session_id, prefix, nlr_level, dbm_fwhm, **kwargs):
    components = get_components_old_pipeline(prefix, nlr_level=nlr_level, dbm_fwhm=dbm_fwhm, i_step=4)
    filename_dbm = Path(SEP_SUFFIX.join(components)).with_suffix(EXT_MINC)
    return check_results_old_pipeline(
        subject_dir=subject_dir,
        session_id=session_id,
        result_filenames=[filename_dbm],
    )

def check_dbm_nii_old_pipeline(subject_dir, session_id, prefix, nlr_level, dbm_fwhm, **kwargs):
    components = get_components_old_pipeline(prefix, nlr_level=nlr_level, dbm_fwhm=dbm_fwhm, i_step=4)
    filename_dbm_nii = Path(SEP_SUFFIX.join(components)).with_suffix(f'{EXT_NIFTI}{EXT_GZIP}')
    return check_results_old_pipeline(
        subject_dir=subject_dir,
        session_id=session_id,
        result_filenames=[filename_dbm_nii],
    )

TRACKER_CONFIGS_OLD_PIPELINE = {
    KW_PIPELINE_COMPLETE: check_dbm_nii_old_pipeline,
    KW_PHASE: {
        "preprocessing": check_preprocessing_old_pipeline,
        "linear_registration": check_linear_registration_old_pipeline,
        "brain_extraction": check_brain_extraction_old_pipeline,
        "nonlinear_registration": check_nonlinear_registration_old_pipeline,
        "dbm": check_dbm_old_pipeline,
        "dbm_nii": check_dbm_nii_old_pipeline,
    }
}

@click.command()
@click.argument('fpath_minc', callback=callback_path)
@click.argument('dpath_out', callback=callback_path)
@click.argument('dpath_qc', callback=callback_path)
@click.argument('prefix_qc')
@click.option('--compress-nii/--no-compress-nii', default=DEFAULT_COMPRESS_NII)
@add_dbm_minc_options()
@add_helper_options()
@with_helper
@check_dbm_inputs
def run_old_from_file(
    helper: ScriptHelper,
    fpath_minc: Path,
    dpath_out: Path,
    dpath_qc: Path,
    prefix_qc: str,
    nlr_level: float,
    dbm_fwhm: float,
    template: str,
    dpath_template: Path,
    fpath_template: Path,
    fpath_template_mask: Path,
    fpath_template_outline: Path,
    dpath_beast_lib: Path,
    fpath_conf: Path,
    compress_nii: bool,
    **kwargs,
):
    def print_skip_message(step_name: str):
        helper.print_info(f'Skipping step: {step_name.upper()}', text_color='yellow')

    def apply_mask(helper: ScriptHelper, fpath_orig, fpath_mask, fpath_out=None, dpath_out=None):
        fpath_orig = Path(fpath_orig)
        if dpath_out is None:
            dpath_out = fpath_orig.parent
        dpath_out = Path(dpath_out)
        if fpath_out is None:
            fpath_out = add_suffix(dpath_out / fpath_orig.name, SUFFIX_MASKED)
        helper.run_command(
            [
                "minccalc",
                "-verbose",
                "-expression",
                "A[0]*A[1]",
                fpath_orig,
                fpath_mask,
                fpath_out,
            ]
        )
        return fpath_out

    # generate fpaths
    fpath_denoised = dpath_out / add_suffix(fpath_minc.name, SUFFIX_DENOISED)
    fpath_norm = add_suffix(fpath_denoised, SUFFIX_NORM)
    fpath_norm_transform = fpath_norm.with_suffix(EXT_TRANSFORM)
    fpath_norm_mask = add_suffix(fpath_norm, SUFFIX_MASK, sep=None)
    fpath_qc_norm = (dpath_qc / add_suffix(prefix_qc, SUFFIX_NORM)).with_suffix(EXT_JPEG)
    fpath_qc_norm_mask = (dpath_qc / add_suffix(prefix_qc, f'{SUFFIX_NORM}{SUFFIX_MASK}')).with_suffix(EXT_JPEG)
    fpath_norm_masked = add_suffix(fpath_norm, SUFFIX_MASKED)
    fpath_nonlinear = add_suffix(fpath_norm_masked, f'{SUFFIX_NONLINEAR}{NLR_LEVEL_PARAM}{int(nlr_level)}')
    fpath_nonlinear_transform = fpath_nonlinear.with_suffix(EXT_TRANSFORM)
    fpath_nonlinear_grid = add_suffix(fpath_nonlinear, SUFFIX_GRID, sep='')
    fpath_qc_nonlinear = (dpath_qc / add_suffix(prefix_qc, SUFFIX_NONLINEAR)).with_suffix(EXT_JPEG)
    fpath_dbm = add_suffix(fpath_nonlinear, f'{SUFFIX_DBM}{DBM_FWHM_PARAM}{int(dbm_fwhm)}')
    fpath_dbm_reshaped = add_suffix(fpath_dbm, SUFFIX_RESHAPED)
    fpath_template_mask_resampled = add_suffix(fpath_template_mask, SUFFIX_RESAMPLED)
    fpath_template_mask_resampled = fpath_dbm_reshaped.parent / fpath_template_mask_resampled.name
    fpath_dbm_masked = add_suffix(fpath_dbm_reshaped, SUFFIX_MASKED)
    fpath_dbm_nii = fpath_dbm_masked.with_suffix(EXT_NIFTI)
    if compress_nii:
        fpath_dbm_nii = Path(f'{fpath_dbm_nii}{EXT_GZIP}')

    # denoise
    if not fpath_denoised.exists():
        helper.run_command(["mincnlm", "-verbose", fpath_minc, fpath_denoised])
    else:
        print_skip_message('denoising')

    # normalize, scale, perform linear registration
    if not (fpath_norm.exists() and fpath_norm_transform.exists()):
        helper.run_command(
            [
                "beast_normalize",
                "-modeldir",
                dpath_template,
                "-modelname",
                template,
                fpath_denoised,
                fpath_norm,
                fpath_norm_transform,
            ]
        )
    else:
        print_skip_message('normalization+scaling+linear registration')

    # get brain mask
    if not fpath_norm_mask.exists():
        helper.run_command(
            [
                "mincbeast",
                "-flip",
                "-fill",
                "-median",
                "-same_resolution",
                "-conf",
                fpath_conf,
                "-verbose",
                dpath_beast_lib,
                fpath_norm,
                fpath_norm_mask,
            ]
        )
    else:
        print_skip_message('brain mask generation')

    # qc linear registration + mask
    if not fpath_qc_norm.exists():
        minc_qc(helper, fpath_norm, fpath_qc_norm, fpath_template_outline, prefix_qc)
    else:
        print_skip_message('QC (normalization+scaling+linear registration)')
    if not fpath_qc_norm_mask.exists():
        minc_qc(helper, fpath_norm, fpath_qc_norm_mask, fpath_norm_mask, prefix_qc)
    else:
        print_skip_message('QC (brain mask generation)')

    # extract brain
    if not fpath_norm_masked.exists():
        fpath_norm_masked = apply_mask(helper, fpath_norm, fpath_norm_mask, fpath_out=fpath_norm_masked)
    else:
        print_skip_message('brain extraction')

    # extract template brain
    fpath_template_masked = apply_mask(
        helper,
        fpath_template,
        fpath_template_mask,
        dpath_out=helper.dpath_tmp,
    )

    # perform nonlinear registration
    if not (fpath_nonlinear.exists() and fpath_nonlinear_transform.exists() and fpath_nonlinear_grid.exists()):
        helper.run_command(
            [
                "nlfit_s",
                "-verbose",
                "-source_mask",
                fpath_norm_mask,
                "-target_mask",
                fpath_template_mask,
                "-level",
                nlr_level,
                fpath_norm,                 # source.mnc
                fpath_template,             # target.mnc
                # fpath_norm_masked,          # source.mnc
                # fpath_template_masked,      # target.mnc
                fpath_nonlinear_transform,  # output.xfm
                fpath_nonlinear,            # output.mnc
            ]
        )
    else:
        print_skip_message('nonlinear registration')

    # qc nonlinear registration
    if not fpath_qc_nonlinear.exists():
        minc_qc(helper, fpath_nonlinear, fpath_qc_nonlinear, fpath_template_outline, prefix_qc)
    else:
        print_skip_message('QC (nonlinear registration)')

    # get DBM map
    if not fpath_dbm.exists():
        helper.run_command(
            [
                "pipeline_dbm.pl",
                "-verbose",
                "--model",
                fpath_template,
                "--fwhm",
                dbm_fwhm,
                fpath_nonlinear_transform,
                fpath_dbm,
            ]
        )
    else:
        print_skip_message('DBM')

    # reshape output before converting to nii to avoid wrong affine
    # need this otherwise nifti file has wrong affine
    # not needed if mincresample is called before
    if not fpath_dbm_reshaped.exists():
        helper.run_command([
            'mincreshape',
            '-dimorder', 'xspace,yspace,zspace',
            fpath_dbm,
            fpath_dbm_reshaped,
        ])
    else:
        print_skip_message('DBM post-processing (reshape)')

    # resample template mask to match DBM map
    if not fpath_template_mask_resampled.exists():
        helper.run_command(
            [
                "mincresample",
                "-like",
                fpath_dbm_reshaped,
                fpath_template_mask,
                fpath_template_mask_resampled,
            ]
        )
    else:
        print_skip_message('DBM post-processing (resample template mask)')

    # apply mask
    if not fpath_dbm_masked.exists():
        apply_mask(helper, fpath_dbm_reshaped, fpath_template_mask_resampled, fpath_out=fpath_dbm_masked)
    else:
        print_skip_message('DBM post-processing (apply mask)')

    # convert back to nifti
    if not fpath_dbm_nii.exists():
        fpath_tmp = fpath_dbm_nii.parent / 'dbm_tmp.nii'
        helper.run_command(["mnc2nii", "-nii", fpath_dbm_masked, fpath_tmp])
        if compress_nii:
            with fpath_dbm_nii.open('wb') as file_gzip:
                helper.run_command(
                    ['gzip', '-c', fpath_tmp], stdout=file_gzip,
                )
        else:
            helper.run_command(
                ['cp', '-fp', fpath_tmp, fpath_dbm_nii]
            )
        helper.run_command(['rm', '-f', fpath_tmp])
    else:
        print_skip_message('DBM post-processing (convert to Nifti)')

    # remove files
    for fpath in [fpath_template_mask_resampled]:
        helper.run_command(['rm', '-f', fpath])

def run_old_from_minc_list(
    helper: ScriptHelper,
    df_minc_list: pd.DataFrame,
    dpath_dbm: Path,
    dpath_out: Path,
    dname_qc: Path,
    dpath_job_logs: Path,
    sge_queue: str,
    template: str = None,
    fpath_conf: Path = None,
    nlr_level: float = None,
    dbm_fwhm: float = None,
    **kwargs,
):
    fpath_command = Path(__file__)
    dpath_src = fpath_command.parent

    # SGE
    varname_job_id = "JOB_ID"
    job_memory = JOB_MEMORY

    fpath_container = dpath_dbm / FNAME_CONTAINER
    if not fpath_container.exists():
        raise FileNotFoundError(f'Container not found: {fpath_container}')

    # flags
    if nlr_level is not None:
        flag_nlr_level = f'--nlr-level {nlr_level}'
    else:
        flag_nlr_level = ''
    if dbm_fwhm is not None:
        flag_dbm_fwhm = f'--dbm-fwhm {dbm_fwhm}'
    else:
        flag_dbm_fwhm = ''

    for subject, session, fpath_minc in df_minc_list.itertuples(index=False):
        
        # make output directory and qc directory
        dpath_subject = dpath_out / subject
        dpath_session_results = dpath_subject / session
        dpath_subject_qc = dpath_subject / dname_qc
        for dpath in [dpath_session_results, dpath_subject_qc]:
            helper.mkdir(dpath, exist_ok=True)

        prefix_qc = f'qc_{subject}_{session}'

        fpath_log = dpath_subject / f'{subject}_{session}.log'
        flag_log = f'--logfile {fpath_log}'

        if template is not None:
            flag_template = f'--template {template}'
        else:
            flag_template = ''
        
        if fpath_conf is not None:
            flag_conf = f'--beast-conf {fpath_conf}'
        else:
            flag_conf = ''

        command_dbm = [
            fpath_command,
            fpath_minc,
            dpath_session_results,
            dpath_subject_qc,
            prefix_qc,
            flag_nlr_level,
            flag_dbm_fwhm,
            flag_template,
            flag_conf,
            flag_log,
        ]

        command_singularity_args = [
            "singularity",
            "run",
            f"--bind {dpath_src}:{dpath_src}",
            f"--bind {fpath_minc}:{fpath_minc}",
            f"--bind {dpath_session_results}:{dpath_session_results}",
            f"--bind {dpath_subject_qc}:{dpath_subject_qc}",
            fpath_container,
        ] + command_dbm
        command_singularity = ' '.join([str(arg) for arg in command_singularity_args])

        command_list = [command_singularity]

        with NamedTemporaryFile("w+t") as file_tmp:
            fpath_submission_tmp = Path(file_tmp.name)

            # job submission script
            varname_command = "COMMAND"
            command = " && ".join(command_list)
            submission_file_lines = [
                "#!/bin/bash",
                (
                    "echo ===================="
                    f" START JOB: ${{{varname_job_id}}} "
                    "===================="
                ),
                "echo `date`",
                f'echo "Memory: {job_memory}"',
                f'{varname_command}="{command}"',
                f"export MKL_NUM_THREADS={N_THREADS_JOBS}",
                f"export NUMEXPR_NUM_THREADS={N_THREADS_JOBS}",
                f"export OMP_NUM_THREADS={N_THREADS_JOBS}",
                'echo "--------------------"',
                f"echo ${{{varname_command}}}",
                'echo "--------------------"',
                f"eval ${{{varname_command}}}",
                "echo `date`",
                (
                    "echo ===================="
                    f" END JOB: ${{{varname_job_id}}} "
                    "===================="
                ),
            ]

            # write to file
            for submission_file_line in submission_file_lines:
                file_tmp.write(f"{submission_file_line}\n")
            file_tmp.flush()  # write right now
            fpath_submission_tmp.chmod(0o744)  # make executable

            # # print file
            # helper.run_command(["cat", fpath_submission_tmp])

            # make logs directory and submit job
            helper.mkdir(dpath_job_logs, exist_ok=True)
            helper.run_command([
                "qsub",
                "-N", f'{PREFIX_PIPELINE}{subject}',
                "-q", sge_queue,
                "-l", f"h_vmem={job_memory}",
                "-j", "y", # join stdout/stderr
                "-o", f"{dpath_job_logs}/$JOB_NAME-$JOB_ID{EXT_LOG}",
                fpath_submission_tmp,
            ])

if __name__ == '__main__':
    run_old_from_file()
