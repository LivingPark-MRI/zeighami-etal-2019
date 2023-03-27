#!/usr/bin/env python
import os
from pathlib import Path
import shutil

import click
import pandas as pd

from bids import BIDSLayout
from bids.layout import parse_file_entities

import livingpark_utils
from livingpark_utils.dataset.ppmi import cohort_id as get_cohort_id
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
DEFAULT_FNAME_ENV = ".env"
DEFAULT_RESET_CACHE = False
DEFAULT_FNAME_CACHE = ".bidslayout"
DEFAULT_DNAME_INPUT = "input"
DEFAULT_DNAME_OUTPUT = "output"
DEFAULT_FNAME_BIDS_LIST = "bids_list-all.csv"
DEFAULT_FNAME_BIDS_LIST_FILTERED = "bids_list.csv"
DEFAULT_FNAME_MINC_LIST = "minc_list.csv"
DEFAULT_FNAME_BAD_SCANS = "bad_scans.csv"
DEFAULT_FNAME_STATUS = "status.csv"
DEFAULT_DNAME_TAR = "tarballs"
DEFAULT_FNAME_TAR = "dbm_results"

# BIDS entities constants
COL_BIDS_SUBJECT = "subject"
COL_BIDS_SESSION = "session"

# init
FNAME_CONTAINER = "nd-minc_1_9_16-fsl_5_0_11-click_livingpark_pandas_pybids.sif" # TODO remove (?)
DNAME_MRI_CODE = "dbm"
FNAME_MRI_CODE = "dbm.py"
FNAME_MRI_SCRIPTS = "scripts"
INIT_DNAME_OUT = "out"
INIT_DNAME_OUT_DBM = "dbm"

# tagged filename patterns
PATTERN_BIDS_LIST_FILTERED = "bids_list-{}.csv"
PATTERN_COHORT = "zeighami-etal-2019-cohort-{}.csv"
PATTERN_MINC_LIST = "minc_list-{}.csv"
PATTERN_TAR = "dbm_results-{}" # without extension
PATTERN_STATUS = "status-{}.csv"

# DBM
COMMAND_PYTHON2 = "python2"

# DBM post
DNAME_VBM = "vbm"
PREFIX_DBM_FILE = "vbm_jac"
FNAME_MASK = f"mask{EXT_MINC}"
SUFFIX_MASKED = "masked"
SUFFIX_TEMPLATE_MASK = "_mask" # for MNI template

# DBM status
TAG_MISSING = "missing"

# tar
FNAME_INFO = "info.csv"

@click.group()
def cli():
    return


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.argument("dpath_bids", type=str, callback=callback_path)
@click.option("--tag", help="tag to differentiate datasets (ex: cohort ID)")
@click.option("--reset-cache/--use-cache", type=bool, default=DEFAULT_RESET_CACHE,
              help=f"Whether to overwrite the pyBIDS database file. Default: {DEFAULT_RESET_CACHE}")
@click.option("--fname-cache", type=str, default=DEFAULT_FNAME_CACHE,
              help=f"Name of pyBIDS database file. Default: {DEFAULT_FNAME_CACHE}")
@add_helper_options()
@with_helper
def bids_list(
    helper: ScriptHelper,
    dpath_dbm: Path,
    dpath_bids: Path,
    tag: str, 
    reset_cache: bool, 
    fname_cache: str, 
):

    # make sure input directory exists
    if not dpath_bids.exists():
        helper.print_error(f"BIDS directory not found: {dpath_bids}")

    # check if file exists
    if tag is None:
        fname_out = DEFAULT_FNAME_BIDS_LIST
    else:
        fname_out = PATTERN_BIDS_LIST_FILTERED.format(tag)
    fpath_out = dpath_dbm / fname_out
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


@cli.command() # TODO refactor arguments/options
@click.argument("dpath_dbm", callback=callback_path)
@click.argument("fpath_cohort", callback=callback_path)
@click.option("--tag", help="tag to differentiate datasets (ex: cohort ID)")
@click.option("--bad-scans", "fname_bad_scans", default=DEFAULT_FNAME_BAD_SCANS,
              help="Path to file listing paths of T1s to ignore (e.g. cases with multiple runs)")
@click.option("--fname_input", default=DEFAULT_FNAME_BIDS_LIST,
              help=f"Name of BIDS list file to filter. Default: {DEFAULT_FNAME_BIDS_LIST}")
@click.option("--subject", "col_cohort_subject", default=COL_PAT_ID,
              help="Name of subject column in cohort file")
@click.option("--session", "col_cohort_session", default=COL_VISIT_TYPE,
              help="Name of session column in cohort file")
@add_helper_options()
@with_helper
def bids_filter(
    helper: ScriptHelper,
    dpath_dbm: Path,
    fpath_cohort: Path,
    tag: str,
    fname_bad_scans: str,
    fname_input: str,
    col_cohort_subject: str,
    col_cohort_session: str,
):

    session_map = {"BL": "1"}

    col_fpath = "fpath"
    cols_merge = [COL_BIDS_SUBJECT, COL_BIDS_SESSION]

    # generate paths
    if tag is None:
        fname_out = DEFAULT_FNAME_BIDS_LIST_FILTERED
    else:
        fname_out = PATTERN_BIDS_LIST_FILTERED.format(tag)
    fpath_out = dpath_dbm / fname_out
    fpath_bids_list = dpath_dbm / fname_input
    fpath_bad_scans = dpath_dbm / fname_bad_scans

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
    cohort_id_original = get_cohort_id(df_cohort)
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
        # helper.echo(','.join(subjects_diff), text_color='yellow') 

    # match by subject and ID
    df_filtered = df_bids_list.merge(df_cohort, on=cols_merge, how="inner")
    helper.print_info(f"Filtered BIDS list:\t\t{df_filtered.shape}")

    if fpath_bad_scans.exists():
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

        fpath_bad_scans = Path(DEFAULT_FNAME_BAD_SCANS)
        while fpath_bad_scans.exists():
            fpath_bad_scans = add_suffix(fpath_bad_scans, sep=None)

        pd.concat(dfs_multiple).to_csv(fpath_bad_scans, header=False, index=False)

        helper.print_error(
            "Found multiple runs for a single session. "
            f"File names written to: {fpath_bad_scans}. "
            "You need to manually check these scans, choose at most one to keep, "
            f"delete it from {fpath_bad_scans}, "
            "then pass that file as input using --bad-scans"
        )

    # print new cohort ID
    new_cohort_id = get_cohort_id(
        df_filtered.drop_duplicates(subset=col_cohort_subject),
    )
    helper.echo(f"COHORT_ID={new_cohort_id}", force_color=False)

    # save
    df_filtered[col_fpath].to_csv(fpath_out, index=False, header=False)
    helper.print_outcome(f"Wrote filtered BIDS list to: {fpath_out}")


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--tag", help="tag to differentiate datasets (ex: cohort ID)")
@click.option("--minc-input-dir", "dname_input", default=DEFAULT_DNAME_INPUT)
@add_silent_option()
@add_helper_options()
@with_helper
@require_minc
def pre_run(
    helper: ScriptHelper,
    dpath_dbm: Path, 
    tag, 
    dname_input,
    silent,
):

    if tag is None:
        fname_bids_list = DEFAULT_FNAME_BIDS_LIST_FILTERED
        fname_out = DEFAULT_FNAME_MINC_LIST
    else:
        fname_bids_list = PATTERN_BIDS_LIST_FILTERED.format(tag)
        fname_out = PATTERN_MINC_LIST.format(tag)

    fpath_bids_list: Path = dpath_dbm / fname_bids_list

    if not fpath_bids_list.exists():
        raise FileNotFoundError()

    dpath_minc_files = dpath_dbm / dname_input
    helper.mkdir(dpath_minc_files, exist_ok=True)
    fpath_out = dpath_dbm / fname_out
    helper.check_file(fpath_out)

    data_input_list = []
    count_skipped = 0
    count_new = 0
    for fpath_nifti in load_list(fpath_bids_list).squeeze("columns").tolist():

        # make sure input file exists and has valid extension
        fpath_nifti = Path(fpath_nifti)
        if not fpath_nifti.exists():
            raise FileNotFoundError(f"Nifti file not found: {fpath_nifti}")
        valid_file_formats = (EXT_NIFTI, f"{EXT_NIFTI}{EXT_GZIP}")
        if not str(fpath_nifti).endswith(valid_file_formats):
            raise RuntimeError(
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

    helper.print_outcome(f"Skipped {count_skipped} files that already existed")
    helper.print_outcome(f"Wrote {count_new} new files to {dpath_minc_files}")

    df_input_list = pd.DataFrame(data_input_list)
    df_input_list.to_csv(fpath_out, header=False, index=False)
    helper.print_outcome(f"Wrote input list to: {fpath_out}")


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--tag", help="tag to differentiate datasets (ex: cohort ID)")
@click.option("--pipeline-dir", "dpath_pipeline", callback=callback_path, required=True,
              help=f"Path to MINC DBM pipeline directory")
@click.option("--template-dir", "dpath_template", callback=callback_path,
              required=True, help=f"Path to MNI template (MINC)")
@click.option("--template", required=True, help=f"MNI template name")
@click.option("--sge/--no-sge", "with_sge", default=True)
@click.option("-q", "--queue", "sge_queue")
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT)
@add_helper_options()
@with_helper
@require_minc
@require_python2
def run(
    helper: ScriptHelper,
    dpath_dbm: Path,
    tag,
    dname_output,
    dpath_pipeline: Path,
    dpath_template: Path,
    template,
    with_sge,
    sge_queue,
):

    dname_nihpd = "nihpd_pipeline"

    if tag is None:
        fname_input_list = DEFAULT_FNAME_MINC_LIST
    else:
        fname_input_list = PATTERN_MINC_LIST.format(tag)
    fpath_input_list = dpath_dbm / fname_input_list
    
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
            f"Make sure to source {fpath_init} before running")
        
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
        "--model-dir", dpath_template,
        "--model-name", template,
        "--lngcls",
        "--denoise",
        "--run",
    ]
    if with_sge:
        command.extend(["--sge", "--queue", sge_queue])
    helper.run_command(command)


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--tag", help="tag to differentiate datasets (ex: cohort ID)")
@click.option("--template-dir", "dpath_template", callback=callback_path,
              required=True, help=f"Path to MNI template (MINC)")
@click.option("--template", required=True, help=f"MNI template name")
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT)
@add_silent_option()
@add_helper_options()
@with_helper
@require_minc
def post_run(
    helper: ScriptHelper, 
    dpath_dbm: Path, 
    tag,
    dpath_template: Path, 
    template,
    dname_output: Path, 
    silent,
):
    
    fpath_mask = add_suffix(dpath_template / template, SUFFIX_TEMPLATE_MASK, sep=None).with_suffix(EXT_MINC)
    if not fpath_mask.exists():
        raise FileNotFoundError(f"Mask file not found: {fpath_mask}")

    if tag is None:
        fname_input_list = DEFAULT_FNAME_MINC_LIST
    else:
        fname_input_list = PATTERN_MINC_LIST.format(tag)
    fpath_input_list = dpath_dbm / fname_input_list

    dpath_output = dpath_dbm / dname_output

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
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--tag", help="tag to differentiate datasets (ex: cohort ID)")
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT)
@click.option("--write-new-list/--no-write-new-list", default=True)
@add_helper_options()
@with_helper
def status(
    helper: ScriptHelper,
    dpath_dbm: Path,
    tag,
    dname_output,
    write_new_list,
):
    
    col_input_t1w = 'input_t1w'

    if tag is None:
        fname_input_list = DEFAULT_FNAME_MINC_LIST
        fname_status = DEFAULT_FNAME_STATUS
    else:
        fname_input_list = PATTERN_MINC_LIST.format(tag)
        fname_status = PATTERN_STATUS.format(tag)
    fpath_input_list = dpath_dbm / fname_input_list
    fpath_status = dpath_dbm / fname_status
    
    helper.check_file(fpath_status)

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
    df_status.drop(columns=col_input_t1w).to_csv(fpath_status, index=False, header=True)
    helper.print_outcome(f"Wrote status file to {fpath_status}")

    if write_new_list:
        fpath_new_list = add_suffix(fpath_input_list, TAG_MISSING) # TODO TAG_MISSING in option (?)
        df_new_list = df_status.loc[
            df_status[KW_PIPELINE_COMPLETE] != SUCCESS,
            [COL_BIDS_SUBJECT, COL_BIDS_SESSION, col_input_t1w],
        ]
        if len(df_new_list) > 0:
            df_new_list.to_csv(fpath_new_list, header=False, index=False)
            helper.print_outcome(f"Wrote missing subjects/sessions to {fpath_new_list}")


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--tag", help="tag to differentiate datasets (ex: cohort ID)")
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT)
@click.option("--tarball-dir", "dname_tar", default=DEFAULT_DNAME_TAR)
@add_silent_option()
@add_helper_options()
@with_helper
def tar(
    helper: ScriptHelper,
    dpath_dbm: Path,
    tag,
    dname_output,
    dname_tar,
    silent,
):
    
    if tag is None:
        fname_status = DEFAULT_FNAME_STATUS
        fname_tar = DEFAULT_FNAME_TAR
    else:
        fname_status = PATTERN_STATUS.format(tag)
        fname_tar = PATTERN_TAR.format(tag)
    fpath_status = dpath_dbm / fname_status

    dpath_output = dpath_dbm / dname_output
    dpath_tmp = helper.dpath_tmp / fname_tar
    dpath_tar = dpath_dbm / dname_tar
    helper.mkdir(dpath_tmp)
    helper.mkdir(dpath_tar, exist_ok=True)

    fpath_out = dpath_tar / f'{fname_tar}{EXT_TAR}{EXT_GZIP}'
    helper.check_file(fpath_out)

    df_status = pd.read_csv(fpath_status, dtype=str)
    df_status_success = df_status.loc[df_status[KW_PIPELINE_COMPLETE] == SUCCESS]
    helper.print_info(f'Tarring {len(df_status_success)} DBM files')

    data_file_info = []
    for subject, session in df_status_success[[COL_BIDS_SUBJECT, COL_BIDS_SESSION]].itertuples(index=False):
        dpath_vbm = dpath_output / subject / session / DNAME_VBM
        fpath_dbm_file = add_suffix(dpath_vbm / f'{PREFIX_DBM_FILE}_{subject}_{session}', SUFFIX_MASKED).with_suffix(f'{EXT_NIFTI}{EXT_GZIP}')

        if not fpath_dbm_file.exists():
            raise RuntimeError(f'File not found: {fpath_dbm_file}')
        
        data_file_info.append({
            COL_BIDS_SUBJECT: subject,
            COL_BIDS_SESSION: session,
            'filename': fpath_dbm_file.name,
        })
        helper.run_command(['ln', '-s', fpath_dbm_file, dpath_tmp], silent=silent)

    # info file
    df_file_info = pd.DataFrame(data_file_info)
    df_file_info.to_csv(dpath_tmp / FNAME_INFO, header=True, index=False)
    helper.print_info(f'Info file name: {FNAME_INFO}')

    # tar command flags:
    #   -h  dereference (tar files instead of symlinks)
    #   -c  create
    #   -z  gzip
    #   -f  name of output file
    #   -C  move to this directory before tarring
    helper.run_command(['tar', '-hczf', fpath_out, '-C', dpath_tmp, '.'], silent=silent)
    helper.print_outcome(f'Wrote file to {fpath_out}')


@cli.command()
@click.argument("dpath-bids", callback=callback_path)
@click.option("-f", "--fname-env", default=DEFAULT_FNAME_ENV)
@add_helper_options()
@with_helper
def init_env(
    dpath_bids: Path,
    fname_env: str,
    helper: ScriptHelper,
):
    dpath_root = Path(__file__).parents[3]

    if helper.verbose:
        helper.print_info(
            f"Generating {fname_env} file:\n"
            f"- Project root directory: {dpath_root}\n"
            f"- BIDS directory: {dpath_bids}"
        )

    # project root directory
    constants = {
        "DPATH_ROOT": dpath_root,
        "DPATH_BIDS": dpath_bids,
    }

    # MRI processing subdirectory
    constants["DPATH_MRI_CODE"] = constants["DPATH_ROOT"] / DNAME_MRI_CODE
    constants["FPATH_MRI_CODE"] = constants["DPATH_MRI_CODE"] / FNAME_MRI_CODE
    constants["FPATH_CONTAINER"] = constants["DPATH_MRI_CODE"] / FNAME_CONTAINER
    constants["DPATH_MRI_SCRIPTS"] = constants["DPATH_MRI_CODE"] / FNAME_MRI_SCRIPTS

    # MRI output
    constants["DPATH_OUT"] = constants["DPATH_ROOT"] / INIT_DNAME_OUT
    constants["DPATH_OUT_DBM"] = constants["DPATH_OUT"] / INIT_DNAME_OUT_DBM
    constants["FPATH_BIDS_LIST"] = constants["DPATH_OUT_DBM"] / DEFAULT_FNAME_BIDS_LIST
    constants["FPATH_BIDS_LIST_FILTERED"] = constants["DPATH_OUT_DBM"] / DEFAULT_FNAME_BIDS_LIST_FILTERED

    if not Path(constants["DPATH_MRI_CODE"]).exists():
        helper.print_error(
            f'Directory not found: {constants["DPATH_MRI_CODE"]}. '
            "Make sure root directory is correct."
        )

    # write dotenv file
    fpath_out = Path(constants["DPATH_MRI_SCRIPTS"], fname_env)
    helper.check_file(fpath_out)
    with fpath_out.open("w") as file_dotenv:
        for key, value in constants.items():
            line = f"{key}={value}\n"
            file_dotenv.write(line)

    helper.print_outcome(f"Variables written to {fpath_out}", text_color="blue")


if __name__ == "__main__":
    cli()
