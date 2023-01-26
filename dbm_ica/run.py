#!/usr/bin/env python
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union

import click
import pandas as pd

from bids import BIDSLayout
from bids.layout import parse_file_entities

import livingpark_utils
from livingpark_utils.zeighamietal.constants import COL_PAT_ID, COL_VISIT_TYPE

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
    SEP_SUFFIX,
    SUFFIX_T1,
)

# output file naming for DBM pipeline
SUFFIX_DENOISED = "denoised"
SUFFIX_NORM = "norm_lr"
SUFFIX_MASK = "_mask"  # binary mask
SUFFIX_MASKED = "masked"  # masked brain
SUFFIX_NONLINEAR = "nlr"
SUFFIX_DBM = "dbm"
SUFFIX_RESHAPED = 'reshaped'
SUFFIX_RESAMPLED = "resampled"

EXT_LOG = ".log"
PREFIX_PIPELINE = "dbm_minc"

# subdirectory names
DNAME_OUTPUT = "output"
DNAME_LOGS = "logs"

# for bids_filter
FNAME_BAD_SCANS = "bad_scans.csv"

# for check_status
FNAME_STATUS = "proc_status.csv"
STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_ALL_PASS = "ALL_PASS"
STATUS_ALL_FAIL = "ALL_FAIL"
STATUS_PARTIAL_PASS = "PARTIAL_PASS"
COL_PROC_PATH = "fpath_input"
COL_SUMMARY = "summary"

# for multi-file command
MIN_I_FILE = 1

# job settings for multi-file command
JOB_TYPE_SLURM = "slurm"
JOB_TYPE_SGE = "sge"
VALID_JOB_TYPES = [JOB_TYPE_SLURM, JOB_TYPE_SGE]
DEFAULT_JOB_MEMORY = "16G"
DEFAULT_JOB_TIME = "0:20:00"
VARNAME_I_FILE = "I_FILE"
BINDPATH_SCRIPTS = "/mnt/scripts"
BINDPATH_BIDS_DATA = "/mnt/bids"
BINDPATH_OUT = "/mnt/out"
BINDPATH_BIDS_LIST = "/mnt/bids_list"
N_THREADS_JOBS = 1#4

# for ICA command
PREFIX_DBM_MERGED = "dbm_merged"
DNAME_MELODIC_RESULTS = "melodic_results"
ICA_RES = 2.0


@click.group()
def cli():
    return


@cli.command()
@click.argument("dpath_bids", type=str, callback=callback_path)
@click.argument("fpath_out", type=str, callback=callback_path)
@add_common_options()
@with_helper
def bids_list(dpath_bids: Path, fpath_out: Path, helper: ScriptHelper):

    # make sure input directory exists
    if not dpath_bids.exists():
        helper.print_error(f"BIDS directory not found: {dpath_bids}")

    # check if file exists
    helper.check_file(fpath_out)

    # create output directory
    helper.mkdir(fpath_out.parent, exist_ok=True)

    # create index for BIDS directory
    bids_layout = BIDSLayout(dpath_bids)

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
            fpath_t1 = Path(fpath_t1).relative_to(dpath_bids)
            file_out.write(f"{fpath_t1}\n")
        helper.print_outcome(f"Wrote BIDS paths to {fpath_out}")


@cli.command()
@click.argument("fpath_bids_list", callback=callback_path)
@click.argument("fpath_cohort", callback=callback_path)
@click.argument("fpath_out", callback=callback_path)
@click.option("--bad-scans", "fpath_bad_scans", callback=callback_path)
@click.option("--subject", "col_cohort_subject", default=COL_PAT_ID)
@click.option("--session", "col_cohort_session", default=COL_VISIT_TYPE)
@add_common_options()
@with_helper
def bids_filter(
    fpath_bids_list: Path,
    fpath_cohort: Path,
    fpath_out: Path,
    fpath_bad_scans: Path,
    col_cohort_subject: str,
    col_cohort_session: str,
    helper: ScriptHelper,
):

    session_map = {"BL": "1"}

    col_fpath = "fpath"
    col_bids_subject = "subject"
    col_bids_session = "session"
    cols_merge = [col_bids_subject, col_bids_session]

    def parse_and_add_path(path, col_path=col_fpath):
        entities = parse_file_entities(path)
        entities[col_path] = path
        return entities

    helper.check_file(fpath_out)

    df_bids_list = pd.DataFrame(
        [
            parse_and_add_path(fpath)
            for fpath in pd.read_csv(fpath_bids_list, header=None)
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

    df_cohort[col_bids_subject] = df_cohort[col_cohort_subject].astype(str)
    df_cohort[col_bids_session] = df_cohort[col_cohort_session].map(session_map)
    if pd.isna(df_cohort[col_bids_session]).any():
        raise RuntimeError(f"Conversion with map {session_map} failed for some rows")

    subjects_all = set(df_bids_list[col_bids_subject])
    subjects_cohort = set(df_cohort[col_bids_subject])
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
            pd.read_csv(fpath_bad_scans, header=None).squeeze("columns").tolist()
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
                (df_filtered[col_bids_subject] == subject)
                & (df_filtered[col_bids_session] == session)
            ]

            dfs_multiple.append(df_multiple[col_fpath])

        fpath_bad_out = Path(FNAME_BAD_SCANS)
        while fpath_bad_out.exists():
            fpath_bad_out = add_suffix(fpath_bad_out, suffix="_", sep=None)

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
@click.argument("dpath_bids", callback=callback_path)
@click.argument("fpath_bids_list", callback=callback_path)
@click.argument("dpath_out", default=".", callback=callback_path)
@click.option("-i", "--i-file", "i_file_single", type=click.IntRange(min=MIN_I_FILE))
@click.option(
    "-r", "--range", "i_file_range", type=click.IntRange(min=MIN_I_FILE), nargs=2
)
@click.option(
    "--job-type", "job_type", type=click.Choice(VALID_JOB_TYPES, case_sensitive=False)
)
@click.option("--job-resource")
@click.option("--job-container", "fpath_container", callback=callback_path)
@click.option("--job-log-dir", "dpath_job_log", callback=callback_path, default=".")
@click.option("--job-memory", default=DEFAULT_JOB_MEMORY)
@click.option("--job-time")
@click.option("--rename-log/--no-rename-log", default=True)
@add_dbm_minc_options()
@add_common_options()
@with_helper
def dbm_from_bids(
    dpath_bids: Path,
    fpath_bids_list: Path,
    dpath_out: Path,
    helper: ScriptHelper,
    i_file_single: Union[int, None],
    i_file_range: Union[tuple, None],
    job_type: str,
    job_resource: str,
    dpath_job_log: Path,
    fpath_container: Union[Path, None],
    job_memory: str,
    job_time: str,
    rename_log: bool,
    **kwargs,
):

    # make output directory now
    # need to mount it when running container
    helper.mkdir(dpath_out, exist_ok=True)

    # convert to range
    if i_file_single is not None:
        i_file_range = (i_file_single, i_file_single)

    # get the maximum possible i_file
    max_i_file = MIN_I_FILE
    with fpath_bids_list.open() as file_bids_list:
        for _ in file_bids_list:
            max_i_file += 1
    max_i_file -= 1  # overcounted

    # get min/max of range
    if i_file_range is not None:
        i_file_start = min(i_file_range)
        i_file_stop = min(max(i_file_range), max_i_file)
    else:
        i_file_start = MIN_I_FILE
        i_file_stop = max_i_file

    # submit job array
    if job_type is not None:

        if job_time is None:
            if job_type != JOB_TYPE_SGE:
                job_time = DEFAULT_JOB_TIME

        # make sure job account/queue is specified
        if job_resource is None:
            helper.print_error(
                "--job-resource must be specified when --job is given",
            )

        # make sure container is specified and exists
        if fpath_container is None:
            helper.print_error(
                "--container must be specified when --job is given",
            )
        if not fpath_container.exists():
            helper.print_error(f"Container not found: {fpath_container}")

        # get path to this file
        fpath_script = Path(__file__).resolve()
        bindpath_script = Path(BINDPATH_SCRIPTS) / fpath_script.name
        dpath_scripts = fpath_script.parent

        script_command_args = [
            f"{bindpath_script} dbm-from-bids",
            f"{BINDPATH_BIDS_DATA} {BINDPATH_BIDS_LIST} {BINDPATH_OUT}",
            f"-i ${VARNAME_I_FILE}",
            "--rename-log" if rename_log else "",
            "--overwrite" if helper.overwrite else "",
        ]

        singularity_command_args = [
            "singularity",
            "run",
            f"--bind {dpath_scripts}:{BINDPATH_SCRIPTS}:ro",
            f"--bind {dpath_bids}:{BINDPATH_BIDS_DATA}:ro",
            f"--bind {fpath_bids_list}:{BINDPATH_BIDS_LIST}:ro",
            f"--bind {dpath_out}:{BINDPATH_OUT}",
            f"{fpath_container}",
            " ".join(script_command_args),
        ]
        singularity_command = " ".join(singularity_command_args)
        command_list = [singularity_command]

        # temporary file for job submission script
        with NamedTemporaryFile("w+t") as file_tmp:

            fpath_submission_tmp = Path(file_tmp.name)

            if job_type == JOB_TYPE_SGE:

                varname_array_job_id = "SGE_TASK_ID"
                varname_job_id = "JOB_ID"

                if (job_time is None) or (len(job_time) == 0):
                    time_flags = []
                else:
                    time_flags = ["-l", f"h_rt={job_time}"]

                job_command_args = [
                    "qsub",
                    "-N",
                    PREFIX_PIPELINE,
                    "-q",
                    job_resource,
                    "-t",
                    f"{i_file_start}-{i_file_stop}:1",
                    "-l",
                    f"h_vmem={job_memory}",
                    # '-l', f'h_rt={job_time}',
                    "-j",
                    "y",
                    "-o",
                    f"{dpath_job_log}/$JOB_NAME-$JOB_ID-$TASK_ID{EXT_LOG}",
                    fpath_submission_tmp,
                ] + time_flags

            elif job_type == JOB_TYPE_SLURM:

                varname_array_job_id = "SLURM_ARRAY_TASK_ID"
                varname_job_id = "SLURM_ARRAY_JOB_ID"
                command_list.insert(0, "module load singularity")

                job_command_args = [
                    "sbatch",
                    f"--account={job_resource}",
                    f"--array={i_file_start}-{i_file_stop}:1",
                    f"--job-name={PREFIX_PIPELINE}",
                    f"--mem={job_memory}",
                    f"--time={job_time}",
                    f"--output={dpath_job_log}/%x-%j-%a{EXT_LOG}",
                    "--open-mode=append",
                    fpath_submission_tmp,
                ]

            else:
                raise NotImplementedError(
                    f"Not implemented for job type {job_type} yet"
                )

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
                f'echo "Time: {job_time}"',
                f"{VARNAME_I_FILE}=${varname_array_job_id}",
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

            # print file
            helper.run_command(["cat", fpath_submission_tmp])

            # make logs directory and submit job
            helper.mkdir(dpath_job_log, exist_ok=True)
            helper.run_command(job_command_args)

    # otherwise run the pipeline directly
    else:

        dpath_results = dpath_out / DNAME_OUTPUT
        dpath_logs = dpath_out / DNAME_LOGS
        helper.mkdir(dpath_results, exist_ok=True)
        layout_results = BIDSLayout(dpath_results, validate=False)

        with fpath_bids_list.open("r") as file_bids_list:

            for i_file, line in enumerate(file_bids_list, start=MIN_I_FILE):

                if i_file < i_file_start:
                    continue
                if i_file > i_file_stop:
                    break

                # remove newline
                fpath_t1_relative = line.strip()

                # skip empty lines
                if fpath_t1_relative == "":
                    continue

                fpath_t1 = dpath_bids / fpath_t1_relative

                # generate path to BIDS-like output directory
                bids_entities = parse_file_entities(fpath_t1)
                dpath_out_bids = Path(layout_results.build_path(bids_entities)).parent

                try:
                    helper.check_dir(dpath_out_bids, prefix=fpath_t1.name.split(".")[0])
                except FileExistsError:
                    helper.print_info(f"Skipping {fpath_t1}")
                    continue

                fpath_log = dpath_logs / f"{PREFIX_PIPELINE}-{i_file}{EXT_LOG}"
                helper.print_info(f"Running pipeline on T1 file {fpath_t1}")
                helper.print_info(f"\tLog: {fpath_log}")

                try:
                    _run_dbm_minc(
                        fpath_nifti=fpath_t1,
                        dpath_out=dpath_out_bids,
                        fpath_log=fpath_log,  # new log file
                        rename_log=rename_log,
                        verbosity=helper.verbosity,
                        quiet=helper.quiet,
                        dry_run=helper.dry_run,
                        overwrite=helper.overwrite,
                        exit_on_error=False,
                        **kwargs,
                    )
                except Exception as ex:
                    helper.print_info(
                        f"An error occured when running on {fpath_t1}: {ex}",
                        text_color="red",
                    )


@cli.command()
@click.argument("fpath_bids_list", callback=callback_path)
@click.argument("dpath_out", callback=callback_path)
@click.option(
    "-s", "--step", "step_suffix_pairs", nargs=2, multiple=True, required=True
)
@click.option("-o", "--file-out", "fname_out", default=FNAME_STATUS)
@click.option("-e", "--extension-t1", "ext_t1", default=f"{EXT_NIFTI}{EXT_GZIP}")
@add_common_options()
@with_helper
def dbm_status(
    helper: ScriptHelper,
    fpath_bids_list: Path,
    dpath_out: Path,
    step_suffix_pairs,
    fname_out,
    ext_t1,
):
    def get_fpath_t1(path_result, dpath_bids_root):
        path_result = Path(path_result)
        dpath_parent_rel = path_result.parent.relative_to(dpath_bids_root)
        prefix = path_result.name.split(".")[0]
        fname_raw = f"{prefix}{ext_t1}"
        return str(dpath_parent_rel / fname_raw)

    col_fpath_t1 = "fpath_t1"

    dpath_bids = dpath_out / DNAME_OUTPUT
    fpath_out = (dpath_out / fname_out).resolve()

    helper.check_file(fpath_out)

    layout = BIDSLayout(dpath_bids, validate=False)
    df_layout = layout.to_df()
    df_layout[col_fpath_t1] = df_layout["path"].apply(
        lambda path: get_fpath_t1(path, dpath_bids)
    )

    helper.print_info("Checking processing status for steps:")
    for step, suffix in step_suffix_pairs:
        if step in [COL_SUMMARY, COL_PROC_PATH]:
            helper.print_error(f"Invalid step name: {step}")
        helper.print_info(f"\t{step}:\t{suffix}")

    t1_proc_status_all = []
    df_t1s = pd.read_csv(fpath_bids_list, header=None, names=[col_fpath_t1])
    n_files = len(df_t1s)
    n_all_pass = 0
    n_partial_pass = 0
    n_fail = 0
    for fpath_t1 in df_t1s[col_fpath_t1]:

        t1_proc_status = layout.parse_file_entities(fpath_t1)
        t1_proc_status[COL_PROC_PATH] = fpath_t1

        df_results = df_layout.loc[df_layout[col_fpath_t1] == fpath_t1]
        extensions = df_results["extension"].tolist()

        n_steps_passed = 0

        for step, suffix in step_suffix_pairs:

            if step in t1_proc_status:
                helper.print_error(f"Invalid step name: {step}")

            if suffix in extensions:
                status = STATUS_PASS
                n_steps_passed += 1
            else:
                status = STATUS_FAIL

            t1_proc_status[step] = status

        if n_steps_passed == 0:
            n_fail += 1
            status_summary = STATUS_ALL_FAIL
        elif n_steps_passed == len(step_suffix_pairs):
            n_all_pass += 1
            status_summary = STATUS_ALL_PASS
        else:
            n_partial_pass += 1
            status_summary = STATUS_PARTIAL_PASS

        t1_proc_status[COL_SUMMARY] = status_summary
        t1_proc_status_all.append(t1_proc_status)

    helper.print_info(f"{n_files} input files total:")
    helper.print_info(f"\t{n_all_pass}\tfiles with all results available")
    helper.print_info(f"\t{n_partial_pass}\tfiles with some results available")
    helper.print_info(f"\t{n_fail}\tfiles with no results available")

    # make df
    df_proc_status = pd.DataFrame(t1_proc_status_all)

    # reorder columns because entities are not the same for all T1 files
    # sometimes there is an additional 'acquisition' field
    # which gets appended to the df, but we want all entities to be before
    # the other columns
    cols_proc = df_proc_status.columns.to_list()
    last_cols = (
        [COL_PROC_PATH] + [step for step, _ in step_suffix_pairs] + [COL_SUMMARY]
    )
    for col in last_cols:
        cols_proc.remove(col)
        cols_proc.append(col)
    df_proc_status = df_proc_status[cols_proc]

    # write file
    df_proc_status.to_csv(fpath_out, index=False, header=True)
    helper.print_outcome(f"Wrote file to {fpath_out}", text_color="blue")


@cli.command()
@click.argument("fpath_nifti", type=str, callback=callback_path)
@click.argument("dpath_out", type=str, default=".", callback=callback_path)
@click.option("--rename-log/--no-rename-log", default=True)
@add_dbm_minc_options()
@add_common_options()
def dbm_from_file(**kwargs):
    _run_dbm_minc(**kwargs)


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.argument("fpath_out", callback=callback_path)
@click.option("-n", type=int, default=None)
@click.option("--suffix", "dbm_suffix")
@click.option("--file-status", "fname_status", default=FNAME_STATUS)
@add_common_options()
@with_helper
def dbm_list(
    helper: ScriptHelper,
    dpath_dbm: Path,
    fpath_out: Path,
    n,
    dbm_suffix,
    fname_status,
):
    def cohort_id_from_filepaths(paths: pd.Series):
        df_subjects = pd.DataFrame(
            {
                COL_PAT_ID: paths.apply(
                    lambda path: int(parse_file_entities(path)["subject"])
                )
            }
        )
        return livingpark_utils.dataset.ppmi.cohort_id(df_subjects)

    if dbm_suffix is None:
        dbm_suffix_components = [
            SUFFIX_DENOISED,
            SUFFIX_NORM,
            SUFFIX_MASKED,
            SUFFIX_NONLINEAR,
            SUFFIX_DBM,
            SUFFIX_RESHAPED,
            SUFFIX_MASKED,
        ]
        dbm_suffix = (
            f"{SEP_SUFFIX}{SEP_SUFFIX.join(dbm_suffix_components)}{EXT_NIFTI}{EXT_GZIP}"
        )

    fpath_status: Path = dpath_dbm / fname_status

    if not fpath_status.exists():
        helper.print_error(f"Processing status file not found: {fpath_status}")

    helper.check_file(fpath_out)

    df_status = pd.read_csv(fpath_status)
    helper.print_info(f"Loaded status file: {df_status.shape}")
    old_cohort_id = cohort_id_from_filepaths(df_status[COL_PROC_PATH])
    helper.print_info(f"\tCohort ID: {old_cohort_id}")

    # only select scans that passed all steps
    df_status_pass = df_status.loc[df_status[COL_SUMMARY] == STATUS_ALL_PASS]

    if n is not None:
        helper.print_info(f"Selecting top {n} files")
        df_status_pass = df_status_pass.iloc[:n]

    helper.print_info(f"Selected {len(df_status_pass)} files")

    dbm_list = df_status_pass[COL_PROC_PATH].apply(
        lambda p: p.split(".")[0] + dbm_suffix
    )

    # print (possibly new) cohort ID
    new_cohort_id = cohort_id_from_filepaths(dbm_list)
    helper.echo(f"COHORT_ID={new_cohort_id}", force_color=False)

    dbm_list.to_csv(fpath_out, header=False, index=False)

    helper.print_outcome(f"Wrote DBM filename list to {fpath_out}")


@cli.command()
@click.argument("fpath_filenames", callback=callback_path)
@click.argument("dpath_dbm", callback=callback_path)
@click.argument("dpath_out", callback=callback_path)
@click.option("--symlink/--no-symlink", default=False)
@click.option("--threshold", type=float, default=3)
@click.option("--resample", "resample_resolution", type=float, default=ICA_RES)
@click.option("--dim", type=int, help="Number of PCA components")
@click.option(
    "--dimest",
    "dim_est_method",
    help="Dimensionality estimation method",
    type=click.Choice(["lap", "bic", "mdl", "aic", "mean"]),
)
@click.option("--shuffle/--no-shuffle", default=False)
@add_common_options()
@with_helper
def ica(
    helper: ScriptHelper,
    fpath_filenames: Path,
    dpath_dbm: Path,
    dpath_out: Path,
    threshold,
    symlink,
    resample_resolution,
    dim,
    dim_est_method,
    shuffle,
    **kwargs,
):

    dpath_tmp = helper.dpath_tmp
    dpath_melodic_results = dpath_out / DNAME_MELODIC_RESULTS
    dpath_dbm_bids = dpath_dbm / DNAME_OUTPUT

    helper.check_dir(dpath_out, prefix=PREFIX_DBM_MERGED)
    helper.mkdir(dpath_out, exist_ok=True)

    # read files and make symlinks (if needed)
    filenames = []
    fpaths_nii_tmp = []
    subjects = []
    with fpath_filenames.open("r") as file_filenames:

        for line in file_filenames:

            line = line.strip()
            if line == "":
                continue

            filenames.append(line)

    if shuffle:
        random.shuffle(filenames)

    for filename in filenames:
        fpath_nii = dpath_dbm_bids / filename
        if not fpath_nii.exists():
            helper.print_error(f"File not found: {fpath_nii}")

        subjects.append(parse_file_entities(fpath_nii)["subject"])

        if symlink:
            fpath_nii_tmp = dpath_tmp / fpath_nii.name
            helper.run_command(
                ["ln", "-s", fpath_nii, fpath_nii_tmp],
                silent=True,
            )
        else:
            fpath_nii_tmp = fpath_nii

        fpaths_nii_tmp.append(fpath_nii_tmp)

    # merge into a single nifti file
    # concatenate in 4th (time) dimension
    fpath_merged = dpath_out / f"{PREFIX_DBM_MERGED}{EXT_NIFTI}{EXT_GZIP}"
    helper.run_command(["fslmerge", "-t", fpath_merged] + fpaths_nii_tmp)

    # check image dimensions
    helper.run_command(["fslinfo", fpath_merged])

    # # downsample
    # fname_resampled = add_suffix(
    #     fpath_merged.name, SUFFIX_RESAMPLED, ext="".join(fpath_merged.suffixes)
    # )
    # fpath_resampled = dpath_out / fname_resampled
    # helper.run_command(
    #     [
    #         "flirt",
    #         "-in",
    #         fpath_merged,
    #         "-ref",
    #         fpath_merged,
    #         "-applyisoxfm",
    #         resample_resolution,
    #         "-nosearch",
    #         "-verbose",
    #         1,
    #         "-out",
    #         fpath_resampled,
    #     ]
    # )

    # check image dimensions
    helper.run_command(["fslinfo", fpath_merged])

    # melodic options
    dim_flag = "" if (dim is None) else f"--dim={dim}"
    dimest_flag = "" if (dim_est_method is None) else f"--dimest={dim_est_method}"

    helper.run_command(
        [
            "melodic",
            "-i",
            fpath_merged,
            "-o",
            dpath_melodic_results,
            dim_flag,  # number of principal components
            dimest_flag,  # dimensionality reduction method
            # '--nobet',          # without brain extraction
            "--Oall",  # output everything
            # '--Ostat',          # thresholded z-statistic maps
            # '--Oorig',          # original ICs (before z-stat)
            "--report",  # create HTML report
            "-v",  # verbose
        ]
    )

    fpath_ICs = dpath_melodic_results / "melodic_IC.nii.gz"
    if not fpath_ICs.exists():
        helper.print_error(f"Cannot find ICA file: {fpath_ICs}")

    fpath_masks = dpath_out / "ICA_masks.nii.gz"
    helper.run_command(
        [
            "fslmaths",
            fpath_ICs,
            "-thr",
            threshold,  # set everything below threshold to 0
            "-bin",  # set everything above 0 to 1
            fpath_masks,
        ]
    )

    # split merged file into component masks
    prefix_mask = "IC_mask"
    dpath_IC_masks = dpath_out / "ICA_masks"
    helper.mkdir(dpath_IC_masks)
    helper.run_command(["fslsplit", fpath_masks, dpath_IC_masks / prefix_mask])

    # find all of the split files
    fpaths_split_mask = list(dpath_IC_masks.glob("*"))
    fpaths_split_mask.sort()

    scores_all = {}
    for i_mask, fpath_mask in enumerate(fpaths_split_mask):

        # fpath_ts = dpath_tmp / f'ts{subject}.txt'
        fpath_dbm_ica = dpath_tmp / f"dbm_IC{i_mask}.txt"
        helper.run_command(
            [
                "fslmeants",
                "-i",
                fpath_merged,  # 4D: X, Y, Z, subject
                "-m",
                fpath_mask,  # 3D: X, Y, Z
                "-o",
                fpath_dbm_ica,  # averages over X/Y/Z for each subject
            ]
        )

        scores = []
        with open(fpath_dbm_ica, "rt") as file_ts:
            for line in file_ts:
                scores.append(float(line.strip()))
        scores_all[f"IC{i_mask+1}"] = scores

    df_scores = pd.DataFrame(data=scores_all, index=subjects)
    helper.print_info(f"df_scores: {df_scores.shape}")
    fpath_scores = dpath_out / "scores.csv"
    df_scores.to_csv(fpath_scores, header=True, index=True)

    helper.run_command(["ls", "-lh", dpath_out])


@with_helper
@check_dbm_inputs
def _run_dbm_minc(
    helper: ScriptHelper,
    fpath_nifti: Path,
    dpath_out: Path,
    dpath_templates: Path,
    template_prefix: str,
    fpath_template: Path,
    fpath_template_mask: Path,
    dpath_beast_lib: Path,
    fpath_conf: Path,
    save_all: bool,
    compress_nii: bool,
    rename_log: bool,
    **kwargs,
):
    def apply_mask(helper: ScriptHelper, fpath_orig, fpath_mask, dpath_out=None, dry_run=False):
        fpath_orig = Path(fpath_orig)
        if dpath_out is None:
            dpath_out = fpath_orig.parent
        dpath_out = Path(dpath_out)
        fpath_out = add_suffix(dpath_out / fpath_orig.name, SUFFIX_MASKED)
        if not dry_run:
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

    def rename_log_callback(helper: ScriptHelper, fpath_new, same_parent=True):
        fpath_old = Path(helper.file_log.name)
        fpath_new = Path(fpath_new)
        if same_parent:
            fpath_new = fpath_old.parent / fpath_new

        def _rename_log_callback():
            helper.run_command(
                ["mv", "-v", fpath_old, fpath_new],
                force=True,
            )

        return _rename_log_callback

    def copy_files_callback(
        helper: ScriptHelper,
        dpath_source: Path,
        dpath_target: Path,
        fpath_main_results: list[Path],
    ):
        def _copy_files_callback():

            # list all result files
            helper.run_command(["ls", "-lh", dpath_source])

            # copy all result files or just the main ones
            if save_all:
                fpaths_to_copy = dpath_source.iterdir()
            else:
                fpaths_to_copy = fpath_main_results

            helper.mkdir(dpath_target, exist_ok=True)

            for fpath_source in fpaths_to_copy:

                # optionally compress nifti files
                if compress_nii and fpath_source.suffix == EXT_NIFTI:
                    fpath_source_gzip = Path(f"{fpath_source}{EXT_GZIP}")
                    with fpath_source_gzip.open("wb") as file_gzip:
                        helper.run_command(
                            ["gzip", "-c", fpath_source], stdout=file_gzip
                        )
                    fpath_source = fpath_source_gzip

                # copy to output directory
                helper.run_command(
                    [
                        "cp",
                        "-vfp",  # verbose, force overwrite, preserve metadata
                        fpath_source,
                        dpath_target,
                    ]
                )

            # list files in output directory
            helper.run_command(["ls", "-lh", dpath_target])

        return _copy_files_callback

    # make sure input file exists and has valid extension
    if not fpath_nifti.exists():
        helper.print_error(f"Nifti file not found: {fpath_nifti}")
    valid_file_formats = (EXT_NIFTI, f"{EXT_NIFTI}{EXT_GZIP}")
    if not str(fpath_nifti).endswith(valid_file_formats):
        helper.print_error(
            f"Invalid file format for {fpath_nifti}. "
            f"Valid extensions are: {valid_file_formats}"
        )

    # skip if output subdirectory already exists and is not empty
    helper.check_dir(dpath_out, prefix=fpath_nifti.name.split(".")[0])

    fpaths_main_results = []
    helper.callbacks_always.append(
        copy_files_callback(
            helper=helper,
            dpath_source=helper.dpath_tmp,
            dpath_target=dpath_out,
            fpath_main_results=fpaths_main_results,
        )
    )

    # if zipped file, unzip
    if fpath_nifti.suffix == EXT_GZIP:
        fpath_raw_nii = helper.dpath_tmp / fpath_nifti.stem  # drop last suffix
        with fpath_raw_nii.open("wb") as file_raw:
            helper.run_command(["zcat", fpath_nifti], stdout=file_raw)
    # else create symbolic link
    else:
        fpath_raw_nii = helper.dpath_tmp / fpath_nifti.name  # keep last suffix
        helper.run_command(["ln", "-s", fpath_nifti, fpath_raw_nii])

    # for renaming the logfile based on nifti file name
    if rename_log and helper.file_log is not None:
        helper.callbacks_always.append(
            rename_log_callback(
                helper=helper,
                fpath_new=f"{fpath_raw_nii.stem}{EXT_LOG}",
                same_parent=True,
            )
        )

    # convert to minc format
    fpath_raw = helper.dpath_tmp / fpath_raw_nii.with_suffix(EXT_MINC)
    helper.run_command(["nii2mnc", fpath_raw_nii, fpath_raw])

    # denoise
    fpath_denoised = add_suffix(fpath_raw, SUFFIX_DENOISED)
    helper.run_command(["mincnlm", "-verbose", fpath_raw, fpath_denoised])
    fpaths_main_results.append(fpath_denoised)

    # normalize, scale, perform linear registration
    fpath_norm = add_suffix(fpath_denoised, SUFFIX_NORM)
    fpath_norm_transform = fpath_norm.with_suffix(EXT_TRANSFORM)
    helper.run_command(
        [
            "beast_normalize",
            "-modeldir",
            dpath_templates,
            "-modelname",
            template_prefix,
            fpath_denoised,
            fpath_norm,
            fpath_norm_transform,
        ]
    )

    # get brain mask
    fpath_mask = add_suffix(fpath_norm, SUFFIX_MASK, sep=SUFFIX_MASK[0])
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
            fpath_mask,
        ]
    )
    fpaths_main_results.append(fpath_mask)

    # extract brain
    fpath_masked = add_suffix(fpath_norm, SUFFIX_MASKED)
    fpath_masked = apply_mask(helper, fpath_norm, fpath_mask)
    fpaths_main_results.append(fpath_masked)

    # extract template brain
    fpath_template_masked = apply_mask(
        helper,
        fpath_template,
        fpath_template_mask,
        dpath_out=helper.dpath_tmp,
    )

    # perform nonlinear registration
    fpath_nonlinear = add_suffix(fpath_masked, SUFFIX_NONLINEAR)
    fpath_nonlinear_transform = fpath_nonlinear.with_suffix(EXT_TRANSFORM)
    helper.run_command(
        [
            "nlfit_s",
            "-verbose",
            "-source_mask",
            fpath_mask,
            "-target_mask",
            fpath_template_mask,
            fpath_masked,
            fpath_template_masked,
            fpath_nonlinear_transform,
            fpath_nonlinear,
        ]
    )
    fpaths_main_results.extend([fpath_nonlinear, fpath_nonlinear_transform])

    # get DBM map
    fpath_dbm = add_suffix(fpath_nonlinear, SUFFIX_DBM)
    helper.run_command(
        [
            "pipeline_dbm.pl",
            "-verbose",
            "--model",
            fpath_template,
            fpath_nonlinear_transform,
            fpath_dbm,
        ]
    )

    # reshape output before converting to nii to avoid wrong affine
    # need this otherwise nifti file has wrong affine
    # not needed if mincresample is called before
    fpath_dbm_reshaped = add_suffix(fpath_dbm, SUFFIX_RESHAPED)
    helper.run_command([
        'mincreshape',
        '-dimorder', 'xspace,yspace,zspace',
        fpath_dbm,
        fpath_dbm_reshaped,
    ])

    # resample template mask to match DBM map
    fpath_template_mask_resampled = add_suffix(fpath_template_mask, SUFFIX_RESAMPLED)
    fpath_template_mask_resampled = fpath_dbm_reshaped.parent / fpath_template_mask_resampled.name
    helper.run_command(
        [
            "mincresample",
            "-like",
            fpath_dbm_reshaped,
            fpath_template_mask,
            fpath_template_mask_resampled,
        ]
    )

    # apply mask
    fpath_dbm_masked = apply_mask(helper, fpath_dbm_reshaped, fpath_template_mask_resampled)

    # convert back to nifti
    fpath_dbm_nii = fpath_dbm_masked.with_suffix(EXT_NIFTI)
    helper.run_command(["mnc2nii", "-nii", fpath_dbm_masked, fpath_dbm_nii])
    fpaths_main_results.append(fpath_dbm_nii)


if __name__ == "__main__":
    cli()
