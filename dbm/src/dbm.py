#!/usr/bin/env python
import os
from pathlib import Path
import shutil

import click
import pandas as pd

from bids import BIDSLayout
from bids.layout import parse_file_entities

from livingpark_utils.dataset.ppmi import cohort_id as get_cohort_id
from livingpark_utils.zeighamietal.constants import COL_PAT_ID, COL_VISIT_TYPE

from dbm_old_pipeline import (
    run_old_from_minc_list, 
    TRACKER_CONFIGS_OLD_PIPELINE,
    QC_FILE_PATTERNS_OLD_PIPELINE,
)

from helpers import (
    add_helper_options,
    add_silent_option,
    add_suffix,
    callback_path,
    check_nihpd_pipeline,
    check_program,
    DEFAULT_NLR_LEVEL,  # for tracking
    DEFAULT_DBM_FWHM,
    DNAME_NIHPD,
    EXT_GZIP,
    EXT_MINC,
    EXT_NIFTI,
    EXT_TAR,
    load_list,
    minc_qc,
    require_minc,
    ScriptHelper,
    SEP_SUFFIX,
    SUFFIX_TEMPLATE_MASK,
    SUFFIX_TEMPLATE_OUTLINE,
    SUFFIX_T1,
    with_helper,
)

from tracker import (
    KW_PHASE, 
    KW_PIPELINE_COMPLETE, 
    TRACKER_CONFIGS, 
    SUCCESS,
)

# default settings
DEFAULT_RESET_CACHE = False
DEFAULT_FNAME_CACHE = ".bidslayout"
DEFAULT_FNAME_ENV = ".env"
DEFAULT_DPATH_PIPELINE=Path("/", "ipl", "quarantine", "experimental", "2013-02-15")
DEFAULT_DPATH_TEMPLATE=Path("/", "ipl", "quarantine", "models", "icbm152_model_09c")
DEFAULT_TEMPLATE="mni_icbm152_t1_tal_nlin_sym_09c"
DEFAULT_SGE_QUEUE="origami.q"
DEFAULT_DNAME_INPUT = "input"
DEFAULT_DNAME_OUTPUT = "output"
DEFAULT_DNAME_TAR = "tarballs"
DEFAULT_FNAME_MINCIGNORE = "mincignore.csv"
DEFAULT_FNAME_MINC_LIST = "minc_list.csv"
DEFAULT_FNAME_STATUS = "status.csv"
DEFAULT_FNAME_TAR = "dbm_results"
DEFAULT_DNAME_QC_OUT = "qc"
DEFAULT_QC_SESSIONS = ["1"]

# BIDS entities constants
COL_BIDS_SUBJECT = "subject"
COL_BIDS_SESSION = "session"

# init
DNAME_SRC = "src"
FNAME_CLI = "dbm.py"
DNAME_SCRIPTS = "scripts"

# tagged filename patterns
PATTERN_BIDS_LIST_FILTERED = "bids_list-{}.csv"
PATTERN_COHORT = "zeighami-etal-2019-cohort-{}.csv"
PATTERN_MINC_LIST = "minc_list-{}.csv"
PATTERN_STATUS = "status-{}.csv"
PATTERN_TAR = "dbm_results-{}" # without extension

# BIDS list
FNAME_BIDS_LIST = "bids_list.csv"

# DBM pre
DNAME_DICOM_INPUT = "dicom"
DNAME_MINC_INPUT = "minc"
DNAME_NIFTI_INPUT = "nifti"
DPATH_DICOM_TO_SUBJECT = Path('PPMI')
COL_IMAGE_ID = "Image ID"
COHORT_SESSION_MAP = {"BL": "1", "U01": "1", "PW": "1", "V04": "1"}
SUFFIX_FROM_NIFTI = "_from_nifti"
SUFFIX_FROM_NIFTI_PPMI = "_from_nifti_ppmi"

# DBM
COMMAND_PYTHON2 = "python2"
SUFFIX_OLD_PIPELINE = "_old_pipeline"
DNAME_JOB_LOGS = "jobs"

# DBM post
DNAME_VBM = "vbm"
DNAME_LINEAR2 = "stx2"
DNAME_NONLINEAR = 'nl'
DNAME_QC = "qc"
PATTERN_DBM_FILE = "vbm_jac_{}_{}.mnc" # subject session
PATTERN_LINEAR2_FILE = "stx2_{}_{}_t1.mnc"
PATTERN_NONLINEAR_TRANSFORM_FILE = 'nl_{}_{}.xfm'
PATTERN_QC_LINEAR2 = "qc_stx2_t1_{}_{}.jpg"
PATTERN_QC_NONLINEAR = "qc_nl_t1_{}_{}.jpg"
FNAME_MASK = f"mask{EXT_MINC}"
SUFFIX_MASKED = "masked"

# DBM status
TAG_MISSING = "missing"

# tar
FNAME_INFO = "info.csv"

# qc
QC_FILE_PATTERNS = {
    "linear": "qc_stx_t1_{}_{}.jpg", # subject session
    "linear_mask": "qc_stx_mask_{}_{}.jpg",
    "linear2": PATTERN_QC_LINEAR2,
    "linear2_mask": "qc_stx2_mask_{}_{}.jpg",
    "nonlinear": PATTERN_QC_NONLINEAR,
}


@click.group()
def cli():
    return


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.argument("dpath_bids", type=str, callback=callback_path)
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
    reset_cache: bool, 
    fname_cache: str, 
):
    """Generate a list of paths to all available T1s in a BIDS dataset.

    Required arguments:
    dpath_dbm : 
        Path to analysis directory
    dpath_bids : 
        Path to BIDS dataset root
    """

    # make sure input directory exists
    if not dpath_bids.exists():
        helper.print_error(f"BIDS directory not found: {dpath_bids}")

    fpath_out = dpath_dbm / FNAME_BIDS_LIST
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
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--cohort-file", 'fpath_cohort', callback=callback_path,
              help="Path to cohort file. Overrides --tag option")
@click.option("--mincignore", "fname_mincignore", default=DEFAULT_FNAME_MINCIGNORE,
              help=f"Name of file containing the list of MINC filenames to ignore. Default: {DEFAULT_FNAME_MINCIGNORE}")
@click.option("--tag", help="Unique tag to differentiate datasets (ex: cohort ID)")
@click.option("--from-nifti/--from-dicom", default=False, 
              help="Convert from NIfTI files instead of DICOMs")
@click.option("--ppmi-nifti/--heudiconv", default=False,
              help=("Use NIfTI files downloaded from PPMI instead of from a BIDS dataset"
                    ". Only applies if --from-nifti is provided"))
@click.option("--input-dir", "dname_input", default=DEFAULT_DNAME_INPUT, 
              help=f"Name of directory containing all input imaging files. Default: {DEFAULT_DNAME_INPUT}")
@click.option("--col-subject", "col_cohort_subject", default=COL_PAT_ID,
              help=f"Name of subject column in cohort file. Default: {COL_PAT_ID}")
@click.option("--col-session", "col_cohort_session", default=COL_VISIT_TYPE,
              help=f"Name of session column in cohort file. Default: {COL_VISIT_TYPE}")
@click.option("--col-image", "col_cohort_image", default=COL_IMAGE_ID,
              help=f"Name of image ID column in cohort file. Default: {COL_IMAGE_ID}")
@click.option("--pipeline-dir", "dpath_pipeline", callback=callback_path,
              default=DEFAULT_DPATH_PIPELINE,
              help=f"Path to MINC DBM pipeline directory. Default: {DEFAULT_DPATH_PIPELINE}")
@add_silent_option()
@add_helper_options()
@with_helper
@require_minc
def pre_run(
    helper: ScriptHelper,
    dpath_dbm: Path,
    fpath_cohort: Path | None,
    fname_mincignore,
    tag,
    from_nifti,
    ppmi_nifti,
    dname_input,
    col_cohort_subject,
    col_cohort_session,
    col_cohort_image,
    dpath_pipeline: Path,
    silent,
):
    """Convert input files to MINC format and ensure there is only one T1 per subject per session.

    Required arguments:
    dpath_dbm :
        Path to analysis directory
    """
    
    def map_session(session):
        try:
            return COHORT_SESSION_MAP[session]
        except KeyError:
            raise RuntimeError(
                f"Conversion with map {COHORT_SESSION_MAP} failed. Missing key: {session}")
        
    def generate_fpath_minc(subject, session, image_id) -> Path:
        dname_minc_session = f'{subject}/{session}'
        fname_converted = image_id
        return Path(dpath_minc / dname_minc_session / fname_converted).with_suffix(EXT_MINC)
    
    def parse_and_add_path(path, col_path):
        entities = parse_file_entities(path)
        entities[col_path] = path
        return entities

    # internal column names (not written)
    col_fpath_nifti = 'fpath_nifti'
    col_fpath_minc = 'fpath_minc'

    # generate paths
    dpath_input: Path = dpath_dbm / dname_input 
    dpath_dicom: Path = dpath_input / DNAME_DICOM_INPUT
    dpath_minc: Path = dpath_input / DNAME_MINC_INPUT
    dpath_nifti = Path(dpath_input) / DNAME_NIFTI_INPUT
    fpath_mincignore: Path = dpath_dbm / fname_mincignore

    if tag is None:
        fname_out = DEFAULT_FNAME_MINC_LIST
        if fpath_cohort is None:
            raise ValueError(f"Either --cohort-file or --tag must be given")
    else:
        fname_out = PATTERN_MINC_LIST.format(tag)
        fpath_cohort = dpath_dbm / PATTERN_COHORT.format(tag)

    if from_nifti:
        if ppmi_nifti:
            fname_out = add_suffix(fname_out, SUFFIX_FROM_NIFTI_PPMI, sep=None)
            dpath_minc = add_suffix(dpath_minc, SUFFIX_FROM_NIFTI_PPMI, sep=None)
        else:
            fname_out = add_suffix(fname_out, SUFFIX_FROM_NIFTI, sep=None)
            dpath_minc = add_suffix(dpath_minc, SUFFIX_FROM_NIFTI, sep=None)

    fpath_bids_list = dpath_dbm / FNAME_BIDS_LIST

    fpath_out = dpath_dbm / fname_out
    helper.check_file(fpath_out)

    df_cohort = pd.read_csv(fpath_cohort, dtype=str)

    # DICOM-to-MINC
    if not from_nifti:

        # require same MINC version as for DBM pipeline
        check_nihpd_pipeline(dpath_pipeline)

        if not dpath_dicom.exists():
            raise FileNotFoundError(f'DICOM directory not found: {dpath_dicom}')
        helper.mkdir(dpath_minc, exist_ok=True)
        
        # convert DICOMs
        count_dicoms_converted = 0
        count_dicoms_ignored = 0
        count_dicoms_skipped = 0
        for dpath_dicom_image, dnames_sub, _ in os.walk(dpath_dicom):
            
            if len(dnames_sub) > 0:
                continue

            image_id = Path(dpath_dicom_image).name
            if image_id.startswith('I'):
                image_id = image_id[1:]

            # check if DICOM Image ID is in the cohort file
            # if not we just ignore it
            try:
                subject, session = df_cohort.set_index(col_cohort_image).loc[
                    image_id, 
                    [col_cohort_subject, col_cohort_session],
                ]
            except KeyError:
                # helper.print_info(f'Ignoring image with ID {image_id}')
                count_dicoms_ignored += 1
                continue

            try:
                int(subject)
            except TypeError:
                raise TypeError(f'Subject IDs must be integers (got: {subject} ({type(subject)}))')

            session = map_session(session)

            fpath_converted = generate_fpath_minc(subject, session, image_id)
            helper.mkdir(fpath_converted.parent, exist_ok=True)

            # do not do anything if the expected new file already exists
            if fpath_converted.exists():
                count_dicoms_skipped += 1
                continue

            # convert
            helper.run_command(
                [
                    'dcm2mnc',
                    '-usecoordinates',
                    '-dname', fpath_converted.parent.relative_to(dpath_minc),
                    '-fname', fpath_converted.with_suffix('').name,
                    dpath_dicom_image,
                    dpath_minc,
                ],
                silent=silent,
            )
            count_dicoms_converted += 1

        helper.print_info(f'Converted {count_dicoms_converted} DICOM directories')
        helper.print_info(f'Skipped {count_dicoms_skipped} DICOM directories that already existed')
        helper.print_info(f'Ignored {count_dicoms_ignored} DICOM directories that were not in cohort file')
    
        # create input list for DBM pipeline
        data_minc_list = []
        missing_image_ids = []
        for subject, session, image_id in df_cohort[[col_cohort_subject, col_cohort_session, col_cohort_image]].itertuples(index=False):
            session = map_session(session)
            fpath_minc = generate_fpath_minc(subject, session, image_id)

            if not fpath_minc.exists():
                missing_image_ids.append(image_id)
                helper.print_info(f'No MINC file found for image {image_id} (subject {subject}, session {session})', text_color='yellow')

            data_minc_list.append({
                col_cohort_subject: int(subject), # needs int for hash to work
                col_cohort_session: session,
                col_fpath_minc: fpath_minc,
            })

        if len(missing_image_ids) > 0:
            raise ValueError(f'Missing {len(missing_image_ids)} images: {",".join(missing_image_ids)}')

        df_minc_list = pd.DataFrame(data_minc_list)

    # NIFTI-to-MINC
    else:

        df_cohort[COL_BIDS_SUBJECT] = df_cohort[col_cohort_subject].astype(str)
        df_cohort[COL_BIDS_SESSION] = df_cohort[col_cohort_session].map(COHORT_SESSION_MAP)

        # load BIDS list
        if not ppmi_nifti:
            df_bids_list = pd.DataFrame(
                [
                    parse_and_add_path(fpath, col_path=col_fpath_nifti)
                    for fpath in load_list(fpath_bids_list)
                    .squeeze("columns")
                    .tolist()
                ]
            )

            helper.print_info(f"Loaded BIDS list:\t\t{df_bids_list.shape}")

            if pd.isna(df_cohort[COL_BIDS_SESSION]).any():
                raise RuntimeError(f"Conversion with map {COHORT_SESSION_MAP} failed for some rows")
            
            subjects_all = set(df_bids_list[COL_BIDS_SUBJECT])
            subjects_cohort = set(df_cohort[COL_BIDS_SUBJECT])
            subjects_diff = subjects_cohort - subjects_all

        else:
            df_bids_list = None

        if ppmi_nifti or len(subjects_diff) > 0:

            if ppmi_nifti:
                # all subjects need to be found in the PPMI download directory
                subjects_diff = set(df_cohort[col_cohort_subject].to_list())
            else:
                helper.echo(f'{len(subjects_diff)} subjects are not in the BIDS list: {",".join(subjects_diff)}', text_color='yellow')

            data_extra_nifti = []
            image_ids_to_download = []
            df_cohort_to_download = df_cohort.loc[df_cohort[col_cohort_subject].isin(subjects_diff), [col_cohort_subject, col_cohort_image, col_cohort_session]]
            for subject, image_id, session in df_cohort_to_download.itertuples(index=False):

                fpaths_nifti = list(dpath_nifti.glob(f'**/*/*{image_id}.nii'))

                if len(fpaths_nifti) == 0:
                    image_ids_to_download.append(image_id)
                elif len(fpaths_nifti) > 1:
                    raise RuntimeError(f'Found more than one file with Image ID: {image_id}')
                else:
                    data_extra_nifti.append({
                        COL_BIDS_SUBJECT: subject,
                        COL_BIDS_SESSION: COHORT_SESSION_MAP[session],
                        col_fpath_nifti: fpaths_nifti[0],
                    })

            if len(image_ids_to_download) > 0:
                raise FileNotFoundError(f'Missing {len(image_ids_to_download)} images: {",".join(image_ids_to_download)}')

            df_extra_nifti = pd.DataFrame(data_extra_nifti)
        else:
            df_extra_nifti = None

        # concatenate lists of NIfTI files from different sources (BIDS/PPMI)
        df_nifti_list: pd.DataFrame = pd.concat([df_bids_list, df_extra_nifti], axis='index')

        # convert to minc
        count_bids_converted = 0
        count_bids_skipped = 0
        data_minc_list = []
        df_nifti_to_convert = df_nifti_list.merge(df_cohort, on=[COL_BIDS_SUBJECT, COL_BIDS_SESSION], how="inner")
        for subject, session, fpath_nifti in df_nifti_to_convert[[COL_BIDS_SUBJECT, COL_BIDS_SESSION, col_fpath_nifti]].itertuples(index=False):
            fpath_nifti = Path(fpath_nifti)
            prefix_nifti = fpath_nifti.name.removesuffix(EXT_GZIP).removesuffix(EXT_NIFTI)
            fpath_minc = generate_fpath_minc(subject, session, prefix_nifti)

            if not Path(fpath_minc).exists():

                # if zipped file, unzip
                if fpath_nifti.suffix == EXT_GZIP:
                    fpath_nifti_unzipped = helper.dpath_tmp / fpath_nifti.stem  # drop last suffix
                    with fpath_nifti_unzipped.open("wb") as file_nifti_unzipped:
                        helper.run_command(["zcat", fpath_nifti], stdout=file_nifti_unzipped, silent=silent)
                # else create symbolic link
                else:
                    fpath_nifti_unzipped = helper.dpath_tmp / fpath_nifti.name  # keep last suffix
                    helper.run_command(["ln", "-s", fpath_nifti, fpath_nifti_unzipped], silent=silent)

                helper.mkdir(Path(fpath_minc).parent, exist_ok=True)
                helper.run_command(
                    [
                        'nii2mnc',
                        fpath_nifti,
                        fpath_minc,
                    ],
                    silent=silent,
                )
                count_bids_converted += 1
            else:
                count_bids_skipped += 1

            data_minc_list.append({
                col_cohort_subject: int(subject), # needs int for hash to work
                col_cohort_session: session,
                col_fpath_minc: fpath_minc,
            })

        helper.print_info(f'Converted {count_bids_converted} Nifti files')
        helper.print_info(f'Skipped {count_bids_skipped} Nifti files that already existed')
    
        df_minc_list = pd.DataFrame(data_minc_list).drop_duplicates()

    # mincignore file is for filtering out specific images
    # so that only 1 image per subject (per session) is used
    if fpath_mincignore.exists():
        fpaths_to_ignore = pd.read_csv(fpath_mincignore, header=None).iloc[:, 0].to_list()
        fpaths_to_ignore = [Path(fpath) for fpath in fpaths_to_ignore]
        helper.print_info(f'Ignoring up to {len(fpaths_to_ignore)} files')
    else:
        fpaths_to_ignore = []

    df_minc_list = df_minc_list.loc[~df_minc_list[col_fpath_minc].isin(fpaths_to_ignore)]

    # check if there are multiple images per subject/session
    counts = df_minc_list.groupby([col_cohort_subject, col_cohort_session])[col_fpath_minc].count()
    with_multiple = counts.loc[counts > 1]

    if len(with_multiple) > 0:
        fpaths_minc_to_check = df_minc_list.set_index([col_cohort_subject, col_cohort_session]).loc[
            with_multiple.index,
        ].reset_index()

        while fpath_mincignore.exists():
            fpath_mincignore = add_suffix(fpath_mincignore, '_', sep=None)

        fpaths_minc_to_check[col_fpath_minc].to_csv(fpath_mincignore, header=False, index=False)

        raise RuntimeError(
            "Found multiple files for a single session for subjects: "
            f"{','.join([str(subject) for subject in fpaths_minc_to_check[col_cohort_subject].drop_duplicates()])}"
            f"\nFile names written to: {fpath_mincignore}. "
            "You need to manually check these scans, choose at most one to keep, "
            f"delete it from {fpath_mincignore}, "
            "then pass that file as input using --mincignore"
        )

    # print new cohort ID (to check if it is the same)
    new_cohort_id = get_cohort_id(
        df_minc_list.drop_duplicates(col_cohort_subject),
    )
    helper.echo(f"COHORT_ID={new_cohort_id}", force_color=False)

    df_minc_list.to_csv(fpath_out, header=False, index=False)
    helper.print_outcome(f"Wrote MINC input list to: {fpath_out}")


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--tag", help="Unique tag to differentiate datasets (ex: cohort ID)")
@click.option("--pipeline-dir", "dpath_pipeline", callback=callback_path,
              default=DEFAULT_DPATH_PIPELINE,
              help=f"Path to MINC DBM pipeline directory. Default: {DEFAULT_DPATH_PIPELINE}")
@click.option("--template-dir", "dpath_template", callback=callback_path,
              default=DEFAULT_DPATH_TEMPLATE,
              help=f"Path to MNI template (MINC). Default: {DEFAULT_DPATH_TEMPLATE}")
@click.option("--template", default=DEFAULT_TEMPLATE,
              help=f"MNI template name. Default: {DEFAULT_TEMPLATE}")
@click.option("--from-nifti/--from-dicom", default=False, 
              help="Convert from NIfTI files instead of DICOMs")
@click.option("--ppmi-nifti/--heudiconv", default=False,
              help=("Use NIfTI files downloaded from PPMI instead of from a BIDS dataset"
                    ". Only applies if --from-nifti is provided"))
@click.option("--old/--no-old", 'use_old_pipeline', default=False,
              help="Use reconstructed pipeline instead of author-provided pipeline")
@click.option("--nlr-level", type=click.FloatRange(min=0.5), default=DEFAULT_NLR_LEVEL,
              help=("Level parameter for nonlinear registration"
                    f". Only used if --old is provided. Default: {DEFAULT_NLR_LEVEL}"))
@click.option("--dbm-fwhm", type=float, default=DEFAULT_DBM_FWHM,
              help=("Blurring kernel for DBM map"
                    f". Only used if --old is provided. Default: {DEFAULT_DBM_FWHM}"))
@click.option("--sge/--no-sge", "with_sge", default=True, help="Use SGE job scheduler (default)")
@click.option("-q", "--queue", "sge_queue", default=DEFAULT_SGE_QUEUE, 
              help=f"Queue name for SGE job scheduler. Default: {DEFAULT_SGE_QUEUE}")
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT,
              help=f"Prefix for output directory name. Default: {DEFAULT_DNAME_OUTPUT}")
@click.option("--qc-dir", "dname_qc", default=DEFAULT_DNAME_QC_OUT,
              help=f"Prefix for QC directory name. Default: {DEFAULT_DNAME_QC_OUT}")
@add_helper_options()
@with_helper
@require_minc
def run(
    helper: ScriptHelper,
    dpath_dbm: Path,
    tag,
    dname_output,
    dpath_pipeline: Path,
    dpath_template: Path,
    template,
    from_nifti,
    ppmi_nifti,
    use_old_pipeline,
    nlr_level,
    dbm_fwhm,
    dname_qc,
    with_sge,
    sge_queue,
):
    """Run the DBM pipeline (using SGE job scheduler by default)

    Required arguments:
    dpath_dbm :
        Path to analysis directory
    """

    # generate paths
    if tag is None:
        fname_minc_list = DEFAULT_FNAME_MINC_LIST
    else:
        fname_minc_list = PATTERN_MINC_LIST.format(tag)

    if from_nifti:
        if ppmi_nifti:
            fname_minc_list = add_suffix(fname_minc_list, SUFFIX_FROM_NIFTI_PPMI, sep=None)
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI_PPMI, sep=None)
        else:
            fname_minc_list = add_suffix(fname_minc_list, SUFFIX_FROM_NIFTI, sep=None)
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI, sep=None)

    if use_old_pipeline:
        dname_output = add_suffix(dname_output, SUFFIX_OLD_PIPELINE, sep=None)

    fpath_minc_list = dpath_dbm / fname_minc_list

    dpath_output = dpath_dbm / dname_output
    helper.mkdir(dpath_output, exist_ok=True)

    # old pipeline (not author-provided)
    if use_old_pipeline:
        helper.print_info('RUNNING OLD PIPELINE', text_color='yellow')
        helper.print_info(f'Using output directory: {dpath_output}')

        if not with_sge:
            raise NotImplementedError("Must use SGE for old pipeline")
        
        dpath_job_logs = dpath_dbm / DNAME_JOB_LOGS
        df_minc_list = pd.read_csv(fpath_minc_list, header=None, dtype=str)
        run_old_from_minc_list(
            helper=helper, 
            df_minc_list=df_minc_list,
            dpath_dbm=dpath_dbm,
            dpath_out=dpath_output,
            dname_qc=dname_qc,
            dpath_job_logs=dpath_job_logs,
            sge_queue=sge_queue,
            template=template,
            nlr_level=nlr_level,
            dbm_fwhm=dbm_fwhm,
        )

    # author-provided pipeline (IPL longitudinal pipeline from Louis Collins lab)
    else:
        check_program("python2", "Python 2")
    
        # validate paths
        fpath_pipeline: Path = dpath_pipeline / DNAME_NIHPD / "python" / "iplLongitudinalPipeline.py"
        fpath_template = Path(dpath_template, template).with_suffix(EXT_MINC)
        for fpath in (fpath_minc_list, fpath_pipeline, fpath_template):
            if not fpath.exists():
                raise RuntimeError(f"File not found: {fpath}")
        
        check_nihpd_pipeline(dpath_pipeline)

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

        command = [
            COMMAND_PYTHON2, fpath_pipeline,
            "--list", fpath_minc_list,
            "--output-dir", dpath_output,
            "--model-dir", dpath_template,
            "--model-name", template,
            "--lngcls",
            "--denoise",
            "--run",
        ]
        if with_sge:
            command.extend(["--sge", "--queue", sge_queue])
        # print(' '.join([str(c) for c in command]))
        helper.run_command(command)


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--tag", help="Unique tag to differentiate datasets (ex: cohort ID)")
@click.option("--template-dir", "dpath_template", callback=callback_path,
              default=DEFAULT_DPATH_TEMPLATE, 
              help=f"Path to MNI template (MINC). Default: {DEFAULT_DPATH_TEMPLATE}")
@click.option("--template", default=DEFAULT_TEMPLATE, 
              help=f"MNI template name. Default: {DEFAULT_TEMPLATE}")
@click.option("--from-nifti/--from-dicom", default=False, 
              help="Convert from NIfTI files instead of DICOMs")
@click.option("--ppmi-nifti/--heudiconv", default=False,
              help=("Use NIfTI files downloaded from PPMI instead of from a BIDS dataset"
                    ". Only applies if --from-nifti is provided"))
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT,
              help=f"Prefix for output directory name. Default: {DEFAULT_DNAME_OUTPUT}")
@click.option("--qc-dir", "dname_qc", default=DEFAULT_DNAME_QC_OUT,
              help=f"Prefix for QC directory name. Default: {DEFAULT_DNAME_QC_OUT}")
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
    from_nifti,
    ppmi_nifti,
    dname_output: Path, 
    with_qc,
    silent,
):
    """Run post-processing after the author-provided pipeline 
    (IPL pipeline from Louis Collins lab). Apply MNI template mask 
    to DBM maps and convert to NIfTI format. Optionally generates QC images
    for the nonlinear registration step.

    Required arguments:
    dpath_dbm :
        Path to analysis directory
    """

    def create_qc_image(fpath, fpath_qc, fpath_mask, title):
        return minc_qc(helper, fpath, fpath_qc, fpath_mask, title, silent=silent)
    
    # generate paths
    fpath_mask = add_suffix(dpath_template / template, SUFFIX_TEMPLATE_MASK, sep=None).with_suffix(EXT_MINC)
    fpath_outline = add_suffix(dpath_template / template, SUFFIX_TEMPLATE_OUTLINE, sep=None).with_suffix(EXT_MINC)
    for fpath in [fpath_mask, fpath_outline]:
        if not fpath.exists():
            raise FileNotFoundError(f"Template file not found: {fpath}")

    if tag is None:
        fname_input_list = DEFAULT_FNAME_MINC_LIST
    else:
        fname_input_list = PATTERN_MINC_LIST.format(tag)

    if from_nifti:
        if ppmi_nifti:
            fname_input_list = add_suffix(fname_input_list, SUFFIX_FROM_NIFTI_PPMI, sep=None)
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI_PPMI, sep=None)
        else:
            fname_input_list = add_suffix(fname_input_list, SUFFIX_FROM_NIFTI, sep=None)
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI, sep=None)
    
    fpath_input_list = dpath_dbm / fname_input_list
    dpath_output = dpath_dbm / dname_output

    # loop over each result set one by one
    count_missing = 0
    count_new = 0
    count_existing = 0
    count_qc = 0
    for subject, session, _ in load_list(fpath_input_list).itertuples(index=False):

        dpath_results = dpath_output / subject / session

        # skip if DBM file does not exist
        fpath_dbm: Path = dpath_results / DNAME_VBM / PATTERN_DBM_FILE.format(subject, session)
        if not fpath_dbm.exists():
            helper.print_info(
                f"DBM file not found for subject {subject}, session {session}.",
                text_color="yellow",
            )
            count_missing += 1
            continue

        fpath_dbm_masked = add_suffix(fpath_dbm, suffix=SUFFIX_MASKED)
        fpath_dbm_masked_nifti = fpath_dbm_masked.with_suffix(EXT_NIFTI)
        
        # post-process if not already done
        if not Path(f"{fpath_dbm_masked_nifti}{EXT_GZIP}").exists():
            fpath_mask_resampled = fpath_dbm.parent / FNAME_MASK

            # resample MNI template mask to match DBM image
            helper.run_command(
                ['mincresample', '-like', fpath_dbm, fpath_mask, fpath_mask_resampled], 
                silent=silent,
            )

            # apply mask
            helper.run_command(
                ['minccalc', '-float', '-expression', 'A[0]*A[1]', fpath_dbm, fpath_mask_resampled, fpath_dbm_masked],
                silent=silent,
            )

            # convert to NIfTI and zip
            helper.run_command(['mnc2nii', fpath_dbm_masked, fpath_dbm_masked_nifti], silent=silent)
            helper.run_command(['gzip', fpath_dbm_masked_nifti], silent=silent)
            count_new += 1
        else:
            count_existing += 1

        # write QC image
        if with_qc:
            fpath_linear2: Path = dpath_results / DNAME_LINEAR2 / PATTERN_LINEAR2_FILE.format(subject, session)
            fpath_nonlinear_transform: Path = dpath_results / DNAME_NONLINEAR / PATTERN_NONLINEAR_TRANSFORM_FILE.format(subject, session)
            fpath_nonlinear = fpath_nonlinear_transform.with_suffix(EXT_MINC)

            dpath_qc: Path = dpath_output / subject / DNAME_QC
            fpath_linear2_qc = dpath_qc / PATTERN_QC_LINEAR2.format(subject, session)
            fpath_nonlinear_qc = dpath_qc / PATTERN_QC_NONLINEAR.format(subject, session)

            # second linear registration
            if fpath_linear2.exists() and not fpath_linear2_qc.exists():
                create_qc_image(
                    fpath_linear2, 
                    fpath_linear2_qc, 
                    fpath_outline, 
                    f'{subject}_{session}',
                )
                count_qc += 1
            
            # nonlinear registration
            if fpath_nonlinear_transform.exists() and not fpath_nonlinear_qc.exists():

                if not fpath_nonlinear.exists():

                    # apply nonlinear transformation
                    helper.run_command(
                        [
                            "mincresample",
                            "-transformation", fpath_nonlinear_transform,
                            "-like", fpath_linear2,
                            fpath_linear2,
                            fpath_nonlinear,
                        ],
                        silent=silent,
                    )

                create_qc_image(
                    fpath_nonlinear, 
                    fpath_nonlinear_qc, 
                    fpath_outline, 
                    f'{subject}_{session}',
                )
                count_qc += 1

    helper.print_outcome(f"Found {count_existing} processed DBM files")
    helper.print_outcome(f"Processed {count_new} new DBM files")
    helper.print_outcome(f"Skipped {count_missing} cases with missing DBM results")
    helper.print_outcome(f"{count_qc} QC images written")


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--tag", help="Unique tag to differentiate datasets (ex: cohort ID)")
@click.option("--from-nifti/--from-dicom", default=False, 
              help="Convert from NIfTI files instead of DICOMs")
@click.option("--ppmi-nifti/--heudiconv", default=False,
              help=("Use NIfTI files downloaded from PPMI instead of from a BIDS dataset"
                    ". Only applies if --from-nifti is provided"))
@click.option("--old/--no-old", 'use_old_pipeline', default=False,
              help="Use reconstructed pipeline instead of author-provided pipeline")
@click.option("--nlr-level", type=click.FloatRange(min=0.5), default=DEFAULT_NLR_LEVEL,
              help=("Level parameter for nonlinear registration"
                    f". Only used if --old is provided. Default: {DEFAULT_NLR_LEVEL}"))
@click.option("--dbm-fwhm", type=float, default=DEFAULT_DBM_FWHM,
              help=("Blurring kernel for DBM map"
                    f". Only used if --old is provided. Default: {DEFAULT_DBM_FWHM}"))
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT,
              help=f"Prefix for output directory name. Default: {DEFAULT_DNAME_OUTPUT}")
@click.option("--write-new-list/--no-write-new-list", default=True, 
              help="Write new input list for failed/missing subjects (default)")
@add_helper_options()
@with_helper
def status(
    helper: ScriptHelper,
    dpath_dbm: Path,
    tag,
    dname_output,
    from_nifti,
    ppmi_nifti,
    use_old_pipeline,
    nlr_level,
    dbm_fwhm,
    write_new_list,
):
    """Check the DBM processing status and write to CSV file.

    Required arguments:
    dpath_dbm :
        Path to analysis directory
    """
    
    col_input_t1w = 'input_t1w'

    # generate paths
    if tag is None:
        fname_input_list = DEFAULT_FNAME_MINC_LIST
        fname_status = DEFAULT_FNAME_STATUS
    else:
        fname_input_list = PATTERN_MINC_LIST.format(tag)
        fname_status = PATTERN_STATUS.format(tag)

    if use_old_pipeline and not (nlr_level == DEFAULT_NLR_LEVEL and dbm_fwhm == DEFAULT_DBM_FWHM):
            fname_status = add_suffix(fname_status, f'_nlr{int(nlr_level)}_dbm{int(dbm_fwhm)}', sep=None)

    if from_nifti:
        if ppmi_nifti:
            fname_input_list = add_suffix(fname_input_list, SUFFIX_FROM_NIFTI_PPMI, sep=None)
            fname_status = add_suffix(fname_status, SUFFIX_FROM_NIFTI_PPMI, sep=None)
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI_PPMI, sep=None)
        else:
            fname_input_list = add_suffix(fname_input_list, SUFFIX_FROM_NIFTI, sep=None)
            fname_status = add_suffix(fname_status, SUFFIX_FROM_NIFTI, sep=None)
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI, sep=None)
    
    # old/author-provided pipelines use different trackers
    if use_old_pipeline:
        fname_status = add_suffix(fname_status, SUFFIX_OLD_PIPELINE, sep=None)
        dname_output = add_suffix(dname_output, SUFFIX_OLD_PIPELINE, sep=None)
        tracker_configs = TRACKER_CONFIGS_OLD_PIPELINE
        kwargs_tracking = {
            'nlr_level': nlr_level,
            'dbm_fwhm': dbm_fwhm,
        }
    else:
        tracker_configs = TRACKER_CONFIGS
        kwargs_tracking = {}

    fpath_input_list = dpath_dbm / fname_input_list
    fpath_status = dpath_dbm / fname_status
    helper.check_file(fpath_status)

    # check status for each input T1 file
    data_status = []
    for subject, session, input_t1w in load_list(fpath_input_list).itertuples(index=False):
    
        kwargs_tracking['prefix'] = Path(input_t1w).stem

        statuses_subject = {
            COL_BIDS_SUBJECT: subject,
            COL_BIDS_SESSION: session,
            col_input_t1w: input_t1w,
        }

        dpath_subject = dpath_dbm / dname_output / subject
        statuses_subject.update({
            phase: phase_func(dpath_subject, session, **kwargs_tracking)
            for phase, phase_func in tracker_configs[KW_PHASE].items()
        })
        statuses_subject[KW_PIPELINE_COMPLETE] = tracker_configs[KW_PIPELINE_COMPLETE](dpath_subject, session, **kwargs_tracking)

        data_status.append(statuses_subject)

    df_status = pd.DataFrame(data_status)
    helper.print_info(df_status[KW_PIPELINE_COMPLETE].value_counts(sort=False, dropna=False))

    # write file
    df_status.drop(columns=col_input_t1w).to_csv(fpath_status, index=False, header=True)
    helper.print_outcome(f"Wrote status file to {fpath_status}")

    # write new input file with missing/failed runs
    if write_new_list:
        fpath_new_list = add_suffix(fpath_input_list, TAG_MISSING)
        if use_old_pipeline:
            fpath_new_list = add_suffix(fpath_new_list, SUFFIX_OLD_PIPELINE, sep=None)
        df_new_list = df_status.loc[
            df_status[KW_PIPELINE_COMPLETE] != SUCCESS,
            [COL_BIDS_SUBJECT, COL_BIDS_SESSION, col_input_t1w],
        ]
        if len(df_new_list) > 0:
            df_new_list.to_csv(fpath_new_list, header=False, index=False)
            helper.print_outcome(f"Wrote missing subjects/sessions to {fpath_new_list}")


@cli.command()
@click.argument("dpath_dbm", callback=callback_path)
@click.option("--tag", help="Unique tag to differentiate datasets (ex: cohort ID)")
@click.option("--from-nifti/--from-dicom", default=False, 
              help="Convert from NIfTI files instead of DICOMs")
@click.option("--ppmi-nifti/--heudiconv", default=False,
              help=("Use NIfTI files downloaded from PPMI instead of from a BIDS dataset"
                    ". Only applies if --from-nifti is provided"))
@click.option("--old/--no-old", 'use_old_pipeline', default=False,
              help="Use reconstructed pipeline instead of author-provided pipeline")
@click.option("--nlr-level", type=click.FloatRange(min=0.5), default=DEFAULT_NLR_LEVEL,
              help=("Level parameter for nonlinear registration"
                    f". Only used if --old is provided. Default: {DEFAULT_NLR_LEVEL}"))
@click.option("--dbm-fwhm", type=float, default=DEFAULT_DBM_FWHM,
              help=("Blurring kernel for DBM map"
                    f". Only used if --old is provided. Default: {DEFAULT_DBM_FWHM}"))
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT,
              help=f"Prefix for output directory name. Default: {DEFAULT_DNAME_OUTPUT}")
@click.option("--tarball-dir", "dname_tar", default=DEFAULT_DNAME_TAR,
              help=f"Name of directory in which to write the tarball. Default: {DEFAULT_DNAME_TAR}")
@add_silent_option()
@add_helper_options()
@with_helper
def tar(
    helper: ScriptHelper,
    dpath_dbm: Path,
    tag,
    from_nifti,
    ppmi_nifti,
    use_old_pipeline,
    nlr_level,
    dbm_fwhm,
    dname_output,
    dname_tar,
    silent,
):
    """Gather all the successfully processed DBM files, tar them
    and move them to another directory.

    Required arguments:
    dpath_dbm :
        Path to analysis directory
    """

    # generate paths
    if from_nifti:
        if ppmi_nifti:
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI_PPMI, sep=None)
        else:
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI, sep=None)
    if use_old_pipeline:
        dname_output = add_suffix(dname_output, SUFFIX_OLD_PIPELINE, sep=None)
    
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

    # read status and get successfully processed scans
    df_status = pd.read_csv(fpath_status, dtype=str)
    df_status_success = df_status.loc[df_status[KW_PIPELINE_COMPLETE] == SUCCESS]
    helper.print_info(f'Tarring {len(df_status_success)} DBM files')

    # gather paths to DBM files to be used in ICA (masked, NIfTI format)
    data_file_info = []
    for subject, session in df_status_success[[COL_BIDS_SUBJECT, COL_BIDS_SESSION]].itertuples(index=False):

        if use_old_pipeline:
            dpath_results_session: Path = dpath_output / subject / session
            fpaths_tmp = [
                fpath 
                for fpath in dpath_results_session.iterdir() 
                if 
                (
                    fpath.suffix in [EXT_NIFTI, EXT_GZIP]
                    and f'nlr_level{int(nlr_level)}{SEP_SUFFIX}dbm_fwhm{int(dbm_fwhm)}' in fpath.name
                )
            ]
            if len(fpaths_tmp) != 1:
                raise RuntimeError(f'Expected exaclty 1 DBM file in {dpath_results_session}, but got: {fpaths_tmp}')
            else:
                fpath_dbm_file = fpaths_tmp[0]
        else:
            dpath_vbm = dpath_output / subject / session / DNAME_VBM
            fpath_dbm_file = add_suffix(dpath_vbm / PATTERN_DBM_FILE.format(subject, session), SUFFIX_MASKED).with_suffix(f'{EXT_NIFTI}{EXT_GZIP}')

        if not fpath_dbm_file.exists():
            raise RuntimeError(f'File not found: {fpath_dbm_file}')
        
        # info file to keep track of which filename goes with which subject/session
        data_file_info.append({
            COL_BIDS_SUBJECT: subject,
            COL_BIDS_SESSION: session,
            'filename': fpath_dbm_file.name,
        })
        helper.run_command(['ln', '-s', fpath_dbm_file, dpath_tmp], silent=silent)

    # write info file
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
@click.argument("dpath_dbm", callback=callback_path)
@click.argument("sessions", nargs=-1) # use default is not provided
@click.option("--from-nifti/--from-dicom", default=False, 
              help="Convert from NIfTI files instead of DICOMs")
@click.option("--ppmi-nifti/--heudiconv", default=False,
              help=("Use NIfTI files downloaded from PPMI instead of from a BIDS dataset"
                    ". Only applies if --from-nifti is provided"))
@click.option("--old/--no-old", 'use_old_pipeline', default=False,
              help="Use reconstructed pipeline instead of author-provided pipeline")
@click.option("--output-dir", "dname_output", default=DEFAULT_DNAME_OUTPUT,
              help=f"Prefix for output directory name. Default: {DEFAULT_DNAME_OUTPUT}")
@click.option("--qc-dir", "dname_qc", default=DEFAULT_DNAME_QC_OUT,
              help=f"Prefix for QC directory name. Default: {DEFAULT_DNAME_QC_OUT}")
@add_helper_options()
@with_helper
def qc(
    helper: ScriptHelper, 
    dpath_dbm: Path, 
    sessions, 
    from_nifti,
    ppmi_nifti,
    use_old_pipeline,
    dname_output, 
    dname_qc, 
):
    """Gather all QC images into a QC directory (flat list for each step of
    the DBM pipeline).

    Required arguments:
    dpath_dbm :
        Path to analysis directory
    """
    
    # generate paths
    if from_nifti:
        if ppmi_nifti:
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI_PPMI, sep=None)
            dname_qc = add_suffix(dname_qc, SUFFIX_FROM_NIFTI_PPMI, sep=None)
        else:
            dname_output = add_suffix(dname_output, SUFFIX_FROM_NIFTI, sep=None)
            dname_qc = add_suffix(dname_qc, SUFFIX_FROM_NIFTI, sep=None)

    # different filename patterns for old/author-provided pipelines
    if use_old_pipeline:
        qc_file_patterns = QC_FILE_PATTERNS_OLD_PIPELINE
        dname_output = add_suffix(dname_output, SUFFIX_OLD_PIPELINE, sep=None)
        dname_qc = add_suffix(dname_qc, SUFFIX_OLD_PIPELINE, sep=None)
    else:
        qc_file_patterns = QC_FILE_PATTERNS

    if len(sessions) == 0:
        sessions = DEFAULT_QC_SESSIONS

    dpath_output = dpath_dbm / dname_output
    dpath_qc_out: Path = dpath_dbm / dname_qc

    helper.mkdir(dpath_qc_out, exist_ok=True)

    # iterate over subjects/sessions and create symlinks
    dpaths_subject = [dpath for dpath in dpath_output.iterdir() if dpath.is_dir()]
    for dpath_subject in dpaths_subject:
        subject = dpath_subject.name

        dpath_subject_qc = dpath_subject / DNAME_QC

        for session in sessions:

            count = 0
            for step, pattern in qc_file_patterns.items():
                fpath_qc = Path(dpath_subject_qc, pattern.format(subject, session))

                if fpath_qc.exists():
                    fpath_qc_link: Path = dpath_qc_out / step / fpath_qc.name
                    helper.mkdir(fpath_qc_link.parent, exist_ok=True)
                    fpath_qc_link.unlink(missing_ok=True)
                    fpath_qc_link.symlink_to(fpath_qc)
                    count += 1

            if count < len(qc_file_patterns):
                helper.print_info(
                    f"Missing {len(qc_file_patterns) - count} QC file(s) "
                    f"for subject {subject}, session {session}"
                )


if __name__ == "__main__":
    cli()
