#!/usr/bin/env python
import os
from pathlib import Path
import shutil

import click
import pandas as pd

from bids import BIDSLayout
from bids.layout import parse_file_entities

import livingpark_utils
from livingpark_utils.zeighamietal.constants import COL_PAT_ID, COL_VISIT_TYPE

from helpers import (
    add_helper_options,
    add_silent_option,
    add_suffix,
    callback_path,
    EXT_GZIP,
    EXT_MINC,
    EXT_NIFTI,
    EXT_TAR,
    load_list,
    require_minc,
    require_python2,
    ScriptHelper,
    SUFFIX_T1,
    with_helper,
)


from tracker import (
    KW_PHASE, 
    KW_PIPELINE_COMPLETE, 
    tracker_configs, 
    SUCCESS,
    UNAVAILABLE,
)

# default settings
DEFAULT_RESET_CACHE = False
DEFAULT_FNAME_CACHE = ".bidslayout"
DEFAULT_DNAME_INPUT = "input"
DEFAULT_DNAME_OUTPUT = "output"
DEFAULT_FNAME_LIST = "list.csv"
DEFAULT_FNAME_BAD_SCANS = "bad_scans.csv"
DEFAULT_FNAME_STATUS = "status.csv"
DEFAULT_DNAME_TAR = "tarballs"
DEFAULT_FNAME_TAR = "dbm_results"
DEFAULT_DPATH_MINC_PIPELINE = Path("/", "ipl", "quarantine", "experimental", "2013-02-15")
DEFAULT_DPATH_TEMPLATE = Path("/", "ipl", "quarantine", "models", "icbm152_model_09c")
DEFAULT_TEMPLATE = "mni_icbm152_t1_tal_nlin_sym_09c"
DEFAULT_SGE_QUEUE = "origami.q"

# BIDS entities constants
COL_BIDS_SUBJECT = "subject"
COL_BIDS_SESSION = "session"

# DBM
COMMAND_PYTHON2 = "python2"

# DBM post
DNAME_VBM = "vbm"
PREFIX_DBM_FILE = "vbm_jac"
FNAME_MASK = f"mask{EXT_MINC}"
SUFFIX_MASKED = "masked"

# DBM status
TAG_MISSING = "missing"

@click.group()
def cli():
    return


@cli.command()
@click.argument("dpath_bids", type=str, callback=callback_path)
@click.argument("fpath_out", type=str, callback=callback_path)
@click.option("--reset-cache/--use-cache", type=bool, default=DEFAULT_RESET_CACHE,
              help=f"Whether to overwrite the pyBIDS database file. Default: {DEFAULT_RESET_CACHE}")
@click.option("--fname-cache", type=str, default=DEFAULT_FNAME_CACHE,
              help=f"Name of pyBIDS database file. Default: {DEFAULT_FNAME_CACHE}")
@add_helper_options()
@with_helper
def bids_list(
    helper: ScriptHelper,
    dpath_bids: Path, 
    fpath_out: Path, 
    reset_cache: bool, 
    fname_cache: str, 
):

    # make sure input directory exists
    if not dpath_bids.exists():
        helper.print_error(f"BIDS directory not found: {dpath_bids}")

    # check if file exists
    helper.check_file(fpath_out)

    # create output directory
    dpath_out = fpath_out.parent
    helper.mkdir(dpath_out, exist_ok=True)

    # create index for BIDS directory
    fpath_cache = dpath_out / fname_cache
    bids_layout = BIDSLayout(dpath_bids, database_path=fpath_cache, reset_database=reset_cache)

    # get all T1 files
    fpaths_t1 = bids_layout.get(
        extension=f"{EXT_NIFTI}{EXT_GZIP}".strip("."),
        suffix=SUFFIX_T1,
        return_type="filename",
    )

    helper.print_info(f"Found {len(fpaths_t1)} T1 files")

    # write paths to output file
    with fpath_out.open("w") as file_out:
        for fpath_t1 in fpaths_t1:
            # fpath_t1 = Path(fpath_t1).relative_to(dpath_bids)
            file_out.write(f"{fpath_t1}\n")
        helper.print_outcome(f"Wrote BIDS paths to {fpath_out}")


@cli.command()
@click.argument("fpath_bids_list", callback=callback_path)
@click.argument("fpath_cohort", callback=callback_path)
@click.argument("fpath_out", callback=callback_path)
@click.option("--bad-scans", "fpath_bad_scans", callback=callback_path,
              help="Path to file listing paths of T1s to ignore (e.g. cases with multiple runs)")
@click.option("--subject", "col_cohort_subject", default=COL_PAT_ID,
              help="Name of subject column in cohort file")
@click.option("--session", "col_cohort_session", default=COL_VISIT_TYPE,
              help="Name of session column in cohort file")
@add_helper_options()
@with_helper
def bids_filter(
    helper: ScriptHelper,
    fpath_bids_list: Path,
    fpath_cohort: Path,
    fpath_out: Path,
    fpath_bad_scans: Path,
    col_cohort_subject: str,
    col_cohort_session: str,
):

    session_map = {"BL": "1"}

    col_fpath = "fpath"
    cols_merge = [COL_BIDS_SUBJECT, COL_BIDS_SESSION]

    def parse_and_add_path(path, col_path=col_fpath):
        entities = parse_file_entities(path)
        entities[col_path] = path
        return entities

    helper.check_file(fpath_out)

    df_bids_list = pd.DataFrame(
        [
            parse_and_add_path(fpath)
            for fpath in load_list(fpath_bids_list)
            .squeeze("columns")
            .tolist()
        ]
    )
    helper.print_info(f"Loaded BIDS list:\t\t{df_bids_list.shape}")

    df_cohort = pd.read_csv(fpath_cohort)
    cohort_id_original = livingpark_utils.dataset.ppmi.cohort_id(df_cohort)
    helper.print_info(
        f"Loaded cohort info:\t\t{df_cohort.shape} (ID={cohort_id_original})"
    )

    df_cohort[COL_BIDS_SUBJECT] = df_cohort[col_cohort_subject].astype(str)
    df_cohort[COL_BIDS_SESSION] = df_cohort[col_cohort_session].map(session_map)
    if pd.isna(df_cohort[COL_BIDS_SESSION]).any():
        raise RuntimeError(f"Conversion with map {session_map} failed for some rows")

    subjects_all = set(df_bids_list[COL_BIDS_SUBJECT])
    subjects_cohort = set(df_cohort[COL_BIDS_SUBJECT])
    subjects_diff = subjects_cohort - subjects_all
    if len(subjects_diff) > 0:
        helper.echo(
            f"{len(subjects_diff)} subjects are not in the BIDS list",
            text_color="yellow",
        )
        # helper.echo(subjects_diff, text_color='yellow') 

    # match by subject and ID
    df_filtered = df_bids_list.merge(df_cohort, on=cols_merge, how="inner")
    helper.print_info(f"Filtered BIDS list:\t\t{df_filtered.shape}")

    if fpath_bad_scans is not None:
        bad_scans = (
            load_list(fpath_bad_scans).squeeze("columns").tolist()
        )
        df_filtered = df_filtered.loc[~df_filtered[col_fpath].isin(bad_scans)]
        helper.print_info(
            f"Removed up to {len(bad_scans)} bad scans:\t{df_filtered.shape}"
        )

    # find duplicate scans
    # go through json sidecar and filter by description
    counts = df_filtered.groupby(cols_merge)[cols_merge[0]].count()
    with_multiple = counts.loc[counts > 1]

    dfs_multiple = []
    if len(with_multiple) > 0:

        for subject, session in with_multiple.index:

            df_multiple = df_filtered.loc[
                (df_filtered[COL_BIDS_SUBJECT] == subject)
                & (df_filtered[COL_BIDS_SESSION] == session)
            ]

            dfs_multiple.append(df_multiple[col_fpath])

        fpath_bad_out = Path(DEFAULT_FNAME_BAD_SCANS)
        while fpath_bad_out.exists():
            fpath_bad_out = add_suffix(fpath_bad_out, sep=None)

        pd.concat(dfs_multiple).to_csv(fpath_bad_out, header=False, index=False)

        helper.print_error(
            "Found multiple runs for a single session. "
            f"File names written to: {fpath_bad_out}. "
            "You need to manually check these scans, choose at most one to keep, "
            f"delete it from {fpath_bad_out}, "
            "then pass that file as input using --bad-scans"
        )

    # print new cohort ID
    new_cohort_id = livingpark_utils.dataset.ppmi.cohort_id(
        df_filtered.drop_duplicates(subset=col_cohort_subject),
    )
    helper.echo(f"COHORT_ID={new_cohort_id}", force_color=False)

    # save
    df_filtered[col_fpath].to_csv(fpath_out, index=False, header=False)
    helper.print_outcome(f"Wrote filtered BIDS list to: {fpath_out}")


@cli.command()
@click.argument("fpath_bids_list", callback=callback_path)
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--minc-input-dir", "dname_input", callback=callback_path, default=DEFAULT_DNAME_INPUT)
@click.option("--outfile", "fname_out", default=DEFAULT_FNAME_LIST)
@add_silent_option()
@add_helper_options()
@with_helper
@require_minc
def pre_run(
    helper: ScriptHelper,
    fpath_bids_list: Path, 
    dpath_dbm: Path, 
    dname_input,
    fname_out,
    silent,
):

    dpath_minc_files = dpath_dbm / dname_input

    fpath_out = dpath_dbm / fname_out
    helper.check_file(fpath_out)

    helper.mkdir(dpath_minc_files, exist_ok=True)

    data_input_list = []
    count_skipped = 0
    count_new = 0
    for fpath_nifti in load_list(fpath_bids_list).squeeze("columns").tolist():

        # make sure input file exists and has valid extension
        fpath_nifti = Path(fpath_nifti)
        if not fpath_nifti.exists():
            helper.print_error(f"Nifti file not found: {fpath_nifti}")
        valid_file_formats = (EXT_NIFTI, f"{EXT_NIFTI}{EXT_GZIP}")
        if not str(fpath_nifti).endswith(valid_file_formats):
            helper.print_error(
                f"Invalid file format for {fpath_nifti}. "
                f"Valid extensions are: {valid_file_formats}"
            )

        fpath_minc: Path = (dpath_minc_files / fpath_nifti.name.strip(EXT_GZIP).strip(EXT_NIFTI)).with_suffix(EXT_MINC)

        # convert to minc format
        if not fpath_minc.exists():

            # if zipped file, unzip
            if fpath_nifti.suffix == EXT_GZIP:
                fpath_nifti_unzipped = helper.dpath_tmp / fpath_nifti.stem  # drop last suffix
                with fpath_nifti_unzipped.open("wb") as file_raw:
                    helper.run_command(["zcat", fpath_nifti], stdout=file_raw, silent=silent)
            # else create symbolic link
            else:
                fpath_nifti_unzipped = helper.dpath_tmp / fpath_nifti.name  # keep last suffix
                helper.run_command(["ln", "-s", fpath_nifti, fpath_nifti_unzipped], silent=silent)

            helper.run_command(["nii2mnc", fpath_nifti_unzipped, fpath_minc], silent=silent)
            count_new += 1
        else:
            count_skipped += 1

        # add subject, session, path to T1
        entities = parse_file_entities(fpath_nifti)
        data_input_list.append([
            entities[COL_BIDS_SUBJECT], 
            entities[COL_BIDS_SESSION], 
            fpath_minc,
        ])

    helper.print_outcome(f"Skipped {count_skipped} files that already existsted")
    helper.print_outcome(f"Wrote {count_new} new files to {dpath_minc_files}")

    df_input_list = pd.DataFrame(data_input_list)
    df_input_list.to_csv(fpath_out, header=False, index=False)
    helper.print_outcome(f"Wrote input list to: {fpath_out}")


@cli.command()
@click.argument("fpath_input_list", callback=callback_path)
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT)
@click.option("--dpath-pipeline", callback=callback_path, 
              default=DEFAULT_DPATH_MINC_PIPELINE,
              help=f"Default: {DEFAULT_DPATH_MINC_PIPELINE}")
@click.option("--dpath-model", "dpath_template", callback=callback_path,
              default=DEFAULT_DPATH_TEMPLATE,
              help=f"Default: {DEFAULT_DPATH_TEMPLATE}")
@click.option("--model-name", "template", default=DEFAULT_TEMPLATE,
              help=f"MNI template name. Default: {DEFAULT_TEMPLATE}")
@click.option("--sge/--no-sge", "with_sge", default=True)
@click.option("-q", "--queue", "sge_queue", default=DEFAULT_SGE_QUEUE)
@add_helper_options()
@with_helper
@require_minc
@require_python2
def run(
    helper: ScriptHelper,
    fpath_input_list,
    dpath_dbm,
    dname_output,
    dpath_pipeline,
    dpath_template,
    template,
    with_sge,
    sge_queue,
):

    dname_nihpd = "nihpd_pipeline"
    
    # validate paths
    fpath_init: Path = dpath_pipeline / "init.sh"
    fpath_pipeline: Path = dpath_pipeline / dname_nihpd / "python" / "iplLongitudinalPipeline.py"
    fpath_template = Path(dpath_template, template).with_suffix(EXT_MINC)
    for fpath in (fpath_input_list, fpath_init, fpath_pipeline, fpath_template):
        if not fpath.exists():
            raise RuntimeError(f"File not found: {fpath}")
    
    pythonpath = os.environ.get('PYTHONPATH')
    if (pythonpath is None) or not dname_nihpd in pythonpath:
        raise RuntimeError(
            "PYTHONPATH environment variable not set correctly. "
            f"Make sure to source {fpath_init} before calling this function")
        
    # need to write a local copy of the MINC script to make sure it uses Python 2
    # otherwise if the user has Python 3 installed it will take precedence
    # and the script will fail
    fpath_pipeline_orig = fpath_pipeline
    fpath_pipeline = dpath_dbm / fpath_pipeline_orig.name
    shutil.copy2(fpath_pipeline_orig, fpath_pipeline)
    with fpath_pipeline.open() as file:
        file_content = file.read()
    file_content = file_content.replace("python", COMMAND_PYTHON2)
    with fpath_pipeline.open('w') as file:
        file.write(file_content)

    dpath_output = dpath_dbm / dname_output
    helper.mkdir(dpath_output, exist_ok=True)

    command = [
        COMMAND_PYTHON2, fpath_pipeline,
        "--list", fpath_input_list,
        "--output-dir", dpath_output,
        "--lngcls",
        "--denoise",
        "--run",
    ]
    if with_sge:
        command.extend(["--sge", "--queue", sge_queue])
    helper.run_command(command)


@cli.command()
@click.argument("fpath_input_list", callback=callback_path)
@click.argument("fpath_mask", callback=callback_path)
@click.argument("dpath_output", callback=callback_path)
@add_silent_option()
@add_helper_options()
@with_helper
@require_minc
def post_run(helper: ScriptHelper, fpath_input_list: Path, fpath_mask: Path, dpath_output: Path, silent):

    count_missing = 0
    count_new = 0
    count_existing = 0
    for subject, session, _ in load_list(fpath_input_list).itertuples(index=False):

        fpath_dbm: Path = dpath_output / subject / session / DNAME_VBM / f"{PREFIX_DBM_FILE}_{subject}_{session}{EXT_MINC}"
        if not fpath_dbm.exists():
            helper.print_info(
                f"DBM file not found for subject {subject}, session {session}.",
                text_color="yellow",
            )
            count_missing += 1
            continue

        fpath_dbm_masked = add_suffix(fpath_dbm, suffix=SUFFIX_MASKED)
        fpath_dbm_masked_nifti = fpath_dbm_masked.with_suffix(EXT_NIFTI)
        
        if not Path(f"{fpath_dbm_masked_nifti}{EXT_GZIP}").exists():
            fpath_mask_resampled = fpath_dbm.parent / FNAME_MASK
            helper.run_command(
                ['mincresample', '-like', fpath_dbm, fpath_mask, fpath_mask_resampled], 
                silent=silent,
            )
            helper.run_command(
                ['minccalc', '-float', '-expression', 'A[0]*A[1]', fpath_dbm, fpath_mask_resampled, fpath_dbm_masked],
                silent=silent,
            )
            helper.run_command(['mnc2nii', fpath_dbm_masked, fpath_dbm_masked_nifti], silent=silent)
            helper.run_command(['gzip', fpath_dbm_masked_nifti], silent=silent)
            count_new += 1
        else:
            count_existing += 1

    helper.print_outcome(f"Found {count_existing} processed DBM files")
    helper.print_outcome(f"Processed {count_new} new DBM files")
    helper.print_outcome(f"Skipped {count_missing} cases with missing DBM results")


@cli.command()
@click.argument("fpath_input_list", callback=callback_path)
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT)
@click.option("--outfile", "fname_out", default=DEFAULT_FNAME_STATUS)
@click.option("--write-new-list/--no-write-new-list", default=True)
@add_helper_options()
@with_helper
def status(
    helper: ScriptHelper,
    fpath_input_list: Path,
    dpath_dbm: Path,
    dname_output,
    fname_out,
    write_new_list,
):
    
    col_input_t1w = 'input_t1w'

    fpath_out = dpath_dbm / fname_out
    helper.check_file(fpath_out)
    
    data_status = []
    for subject, session, input_t1w in load_list(fpath_input_list).itertuples(index=False):
    
        statuses_subject = {
            COL_BIDS_SUBJECT: subject,
            COL_BIDS_SESSION: session,
            col_input_t1w: input_t1w,
        }

        dpath_subject = dpath_dbm / dname_output / subject
        statuses_subject.update({
            phase: phase_func(dpath_subject, session)
            for phase, phase_func in tracker_configs[KW_PHASE].items()
        })
        statuses_subject[KW_PIPELINE_COMPLETE] = tracker_configs[KW_PIPELINE_COMPLETE](dpath_subject, session)

        data_status.append(statuses_subject)

    df_status = pd.DataFrame(data_status)
    helper.print_info(df_status[KW_PIPELINE_COMPLETE].value_counts(sort=False, dropna=False))

    # write file
    df_status.drop(columns=col_input_t1w).to_csv(fpath_out, index=False, header=True)
    helper.print_outcome(f"Wrote status file to {fpath_out}")

    if write_new_list:
        fpath_new_list = add_suffix(fpath_input_list, TAG_MISSING)
        df_new_list = df_status.loc[
            df_status[KW_PIPELINE_COMPLETE] == UNAVAILABLE,
            [COL_BIDS_SUBJECT, COL_BIDS_SESSION, col_input_t1w],
        ]
        if len(df_new_list) > 0:
            df_new_list.to_csv(fpath_new_list, header=False, index=False)
            helper.print_outcome(f"Wrote missing subjects/sessions to {fpath_new_list}")


@cli.command()
@click.argument("fpath_status", callback=callback_path)
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT)
@click.option("--tarball-dir", "dname_tar", default=DEFAULT_DNAME_TAR)
@click.option("--outfile", "fname_out", default=DEFAULT_FNAME_TAR)
@add_silent_option()
@add_helper_options()
@with_helper
def tar(
    helper: ScriptHelper,
    fpath_status: Path,
    dpath_dbm: Path,
    dname_output,
    dname_tar,
    fname_out,
    silent,
):
    dpath_output = dpath_dbm / dname_output
    dpath_tmp = helper.dpath_tmp / fname_out
    dpath_tar = dpath_dbm / dname_tar
    helper.mkdir(dpath_tmp)
    helper.mkdir(dpath_tar, exist_ok=True)

    fpath_out = dpath_tar / f'{fname_out}{EXT_TAR}{EXT_GZIP}'
    helper.check_file(fpath_out)

    df_status = pd.read_csv(fpath_status, dtype=str)
    df_status_success = df_status.loc[df_status[KW_PIPELINE_COMPLETE] == SUCCESS]

    for subject, session in df_status_success[[COL_BIDS_SUBJECT, COL_BIDS_SESSION]].itertuples(index=False):
        dpath_vbm = dpath_output / subject / session / DNAME_VBM
        fpath_dbm_file = add_suffix(dpath_vbm / f'{PREFIX_DBM_FILE}_{subject}_{session}', SUFFIX_MASKED).with_suffix(f'{EXT_NIFTI}{EXT_GZIP}')

        if not fpath_dbm_file.exists():
            raise RuntimeError(f'File not found: {fpath_dbm_file}')
        
        helper.run_command(['ln', '-s', fpath_dbm_file, dpath_tmp], silent=silent)

    # tar command flags:
    #   -h  dereference (tar files instead of symlinks)
    #   -c  create
    #   -z  gzip
    #   -f  output filename
    #   -C  move to this directory before tarring
    helper.run_command(['tar', '-hczf', fpath_out, '-C', dpath_tmp, '.'], silent=silent)

    helper.print_outcome(f'Wrote file to {fpath_out}')


if __name__ == "__main__":
    cli()
