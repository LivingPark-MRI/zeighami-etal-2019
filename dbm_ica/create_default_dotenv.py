#!/usr/bin/env python
from pathlib import Path

import click
from helpers import (
    add_common_options,
    add_suffix,
    callback_path,
    ScriptHelper,
    with_helper,
)

FNAME_CONTAINER = "nd-minc_1_9_16-fsl_5_0_11-click_livingpark_pandas_pybids.sif"

FNAME_BIDS_LIST = "bids_list.txt"
FNAME_DBM_LIST = "dbm_list.txt"
TAG_BIDS_LIST_ALL = "all"

DNAME_OUT = "out"
DNAME_OUT_DBM = "dbm"
DNAME_OUT_ICA = "ica"

DNAME_JOB_LOGS = "jobs"


@click.command()
@click.argument("dpath-root", default=".", callback=callback_path)
@click.argument("dpath-bids", default="bids", callback=callback_path)
@click.option(
    "-j",
    "--job",
    "job_type_and_resource",
    nargs=2,
    help="Job submission system and account/queue",
)
@click.option("-f", "--fname-dotenv", default=".env")
@add_common_options()
@with_helper
def create_default_dotenv(
    dpath_root: Path,
    dpath_bids: Path,
    job_type_and_resource: tuple,
    fname_dotenv: str,
    helper: ScriptHelper,
):

    if job_type_and_resource is None:
        job_type_and_resource = ("", "")

    job_type, job_resource = job_type_and_resource

    if helper.verbose:
        helper.print_info(
            f"Generating default dotenv file with root directory: {dpath_root}"
        )

    # project root directory
    constants = {
        "DPATH_ROOT": dpath_root,
        "DPATH_BIDS": dpath_bids,
        "JOB_TYPE": job_type,
        "JOB_RESOURCE": job_resource,
    }

    # MRI processing subdirectory
    constants["DPATH_MRI"] = constants["DPATH_ROOT"] / "dbm_ica"
    constants["FPATH_SCRIPT"] = constants["DPATH_MRI"] / "run.py"
    constants["FPATH_CONTAINER"] = constants["DPATH_MRI"] / FNAME_CONTAINER
    constants["DPATH_JOB_LOGS"] = constants["DPATH_ROOT"] / DNAME_JOB_LOGS

    # MRI output
    constants["DPATH_OUT"] = constants["DPATH_ROOT"] / DNAME_OUT
    constants["DPATH_OUT_DBM"] = constants["DPATH_OUT"] / DNAME_OUT_DBM
    constants["FPATH_BIDS_LIST"] = constants["DPATH_OUT_DBM"] / FNAME_BIDS_LIST
    constants["FPATH_BIDS_LIST_ALL"] = add_suffix(
        constants["FPATH_BIDS_LIST"], TAG_BIDS_LIST_ALL, sep="-"
    )
    constants["DPATH_OUT_ICA"] = constants["DPATH_OUT"] / DNAME_OUT_ICA
    constants["FPATH_DBM_LIST"] = constants["DPATH_OUT_ICA"] / FNAME_DBM_LIST

    if not Path(constants["DPATH_MRI"]).exists():
        helper.print_error(
            f'Directory not found: {constants["DPATH_MRI"]}. '
            "Make sure root directory is correct."
        )

    # write dotenv file
    fpath_out = Path(constants["DPATH_MRI"], fname_dotenv)
    helper.check_file(fpath_out)
    with fpath_out.open("w") as file_dotenv:
        for key, value in constants.items():
            line = f"{key}={value}\n"
            file_dotenv.write(line)

    helper.print_outcome(f"Variables written to {fpath_out}", text_color="blue")


if __name__ == "__main__":
    create_default_dotenv()
