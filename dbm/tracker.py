import os
from pathlib import Path
from typing import Iterable

from helpers import EXT_GZIP, EXT_MINC, EXT_NIFTI, EXT_TRANSFORM

# config keywords
KW_PIPELINE_COMPLETE = "pipeline_complete"
KW_PHASE = "Phase_"

# status flags
SUCCESS = "SUCCESS"
FAIL = "FAIL"
INCOMPLETE = "INCOMPLETE"
UNAVAILABLE = "UNAVAILABLE"

# file suffixes
SUFFIX_MASK = f"_mask{EXT_MINC}"
SUFFIX_T1_MNC = f"_t1{EXT_MINC}"
SUFFIX_T1_XFM = f"_t1{EXT_TRANSFORM}"
SUFFIX_GRID = f"_grid_0{EXT_MINC}"
SUFFIX_DBM_NII = f"-masked{EXT_NIFTI}{EXT_GZIP}"

def build_filenames(subject_dir, session_id, results_subdir, patterns: Iterable[str]):
    if isinstance(patterns, (os.PathLike, str)):
        patterns = [patterns]
    subject_id = Path(subject_dir).name

    results_dir = Path(subject_dir, session_id)
    result_filenames = [
        Path(results_subdir, pattern.format(subject_id, session_id))
        for pattern in patterns
    ]
    return results_dir, result_filenames

def check_files(results_dir, result_filenames):
    if isinstance(result_filenames, (os.PathLike, str)):
        result_filenames = [result_filenames]
    if len(result_filenames) == 0:
        raise ValueError(
            f"result_files must be a path or a (non-empty) list of paths"
        )
    
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return UNAVAILABLE
    
    statuses = set()
    for result_filename in result_filenames:
        if (results_dir / result_filename).exists():
            statuses.add(SUCCESS)
        else:
            statuses.add(FAIL)
    
    if len(statuses) == 1:
        return statuses.pop()
    
    else:
        return INCOMPLETE
    
def check_results(subject_dir, session_id, results_subdir, patterns: Iterable[str]):
    results_dir, result_filenames = build_filenames(
        subject_dir=subject_dir, 
        session_id=session_id, 
        results_subdir=results_subdir,
        patterns=patterns,
    )
    return check_files(results_dir=results_dir, result_filenames=result_filenames)

def check_preprocessing(subject_dir, session_id):
    return check_results(
        subject_dir=subject_dir,
        session_id=session_id,
        results_subdir="clp",
        patterns=[
            "clp_{}_{}" + SUFFIX_MASK,
            "clp_{}_{}" + SUFFIX_T1_MNC,
        ])
    
def check_linear_registration1(subject_dir, session_id):
    return check_results(
        subject_dir=subject_dir,
        session_id=session_id,
        results_subdir="stx",
        patterns=[
            "stx_{}_{}" + SUFFIX_MASK,
            "stx_{}_{}" + SUFFIX_T1_MNC,
            "stx_{}_{}" + SUFFIX_T1_XFM,
            "nsstx_{}_{}" + SUFFIX_MASK,
            "nsstx_{}_{}" + SUFFIX_T1_MNC,
            "nsstx_{}_{}" + SUFFIX_T1_XFM,
        ])


def check_linear_registration2(subject_dir, session_id):
    return check_results(
        subject_dir=subject_dir,
        session_id=session_id,
        results_subdir="stx2",
        patterns=[
            "stx2_{}_{}" + SUFFIX_MASK,
            "stx2_{}_{}" + SUFFIX_T1_MNC,
            "stx2_{}_{}" + SUFFIX_T1_XFM,
        ])

def check_nonlinear_registration(subject_dir, session_id):
    return check_results(
        subject_dir=subject_dir,
        session_id=session_id,
        results_subdir="nl",
        patterns=[
            "nl_{}_{}" + SUFFIX_GRID,
            "nl_{}_{}" + EXT_TRANSFORM,
        ])

def check_dbm(subject_dir, session_id):
    return check_results(
        subject_dir=subject_dir,
        session_id=session_id,
        results_subdir="vbm",
        patterns=[
            "vbm_jac_{}_{}" + EXT_MINC,
        ])

def check_dbm_nii(subject_dir, session_id):
    return check_results(
        subject_dir=subject_dir,
        session_id=session_id,
        results_subdir="vbm",
        patterns=[
            "vbm_jac_{}_{}" + EXT_MINC,
            "vbm_jac_{}_{}" + SUFFIX_DBM_NII,
        ])

tracker_configs = {
    KW_PIPELINE_COMPLETE: check_dbm_nii,
    KW_PHASE: {
        "preprocessing": check_preprocessing,
        "linear_registration1": check_linear_registration1,
        "linear_registration2": check_linear_registration2,
        "nonlinear_registration": check_nonlinear_registration,
        "dbm": check_dbm,
        "dbm_nii": check_dbm_nii,
    }
}
