#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
import traceback

from contextlib import nullcontext
from functools import wraps
from pathlib import Path
from typing import Union

import click

DEFAULT_VERBOSITY = 2
PREFIX_RUN = '[RUN] '
PREFIX_ERROR = '[ERROR] '

EXT_NIFTI = '.nii'
EXT_GZIP = '.gz'
EXT_MINC = '.mnc'
EXT_TRANSFORM = '.xfm'
SUFFIX_T1 = 'T1w'

SUFFIX_TEMPLATE_MASK = '_mask' # MNI template naming convention
ENV_VAR_DPATH_SHARE = 'MNI_DATAPATH'
DEFAULT_BEAST_CONF = 'default.1mm.conf'
DEFAULT_TEMPLATE = 'mni_icbm152_t1_tal_nlin_sym_09c'
DNAME_BEAST_LIB = 'beast-library-1.1'
DNAME_TEMPLATE_MAP = {
    'mni_icbm152_t1_tal_nlin_sym_09c': 'icbm152_model_09c',
    'mni_icbm152_t1_tal_nlin_sym_09a': 'icbm152_model_09a',
}

def add_suffix(
    path: Union[Path, str], 
    suffix: str, 
    sep: Union[str, None] = '.',
) -> Path:
    if sep is not None:
        if suffix.startswith(sep):
            suffix = suffix[len(sep):]
    else:
        sep = ''
    path = Path(path)
    return path.parent / f'{path.stem}{sep}{suffix}{path.suffix}'

def process_path(path: str) -> Path:
    return Path(path).expanduser().absolute()

def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func
    return _add_options

def add_common_options():
    common_options = [
        click.option('--logfile', 'fpath_log', type=str, callback=callback_path,
                help='Path to log file'),
        click.option('--overwrite/--no-overwrite', default=False,
                help='Overwrite existing result files.'),
        click.option('--dry-run/--no-dry-run', default=False,
                    help='Print shell commands without executing them.'),
        click.option('-v', '--verbose', 'verbosity', count=True, 
                    default=DEFAULT_VERBOSITY,
                    help='Set/increase verbosity level (cumulative). '
                        f'Default level: {DEFAULT_VERBOSITY}.'),
        click.option('--quiet', is_flag=True, default=False,
                    help='Suppress output whenever possible. '
                        'Has priority over -v/--verbose flags.'),
    ]
    return add_options(common_options)

def add_dbm_minc_options():
    dbm_minc_options = [
        click.option('--share-dir', 'dpath_share', 
              callback=callback_path, envvar=ENV_VAR_DPATH_SHARE,
              help='Path to directory containing BEaST library and '
                   f'anatomical models. Uses ${ENV_VAR_DPATH_SHARE} '
                   'environment variable if not specified.'),
        click.option('--template-dir', 'dpath_templates', callback=callback_path,
                    help='Directory containing anatomical templates.'),
        click.option('--template', 'template_prefix', default=DEFAULT_TEMPLATE,
                     help='Prefix for anatomical model files. '
                          f'Valid names: {list(DNAME_TEMPLATE_MAP.keys())}. '
                          f'Default: {DEFAULT_TEMPLATE}.'),
        click.option('--beast-lib-dir', 'dpath_beast_lib', callback=callback_path,
                     help='Path to library directory for mincbeast.'),
        click.option('--beast-conf', default=DEFAULT_BEAST_CONF,
                     help='Name of configuration file for mincbeast. '
                          f'Default: {DEFAULT_BEAST_CONF}.'),
        click.option('--save-all/--save-subset', default=True,
                     help='Save all intermediate files')
    ]
    return add_options(dbm_minc_options)

def callback_path(ctx, param, value):
    if value is None:
        return None
    return process_path(value)

def with_helper(func):
    @wraps(func)
    def _with_helper(
        fpath_log: Path = None, 
        verbosity: int = DEFAULT_VERBOSITY,
        quiet: bool = False,
        dry_run: bool = False,
        overwrite: bool = False,
        prefix_run: str = PREFIX_RUN,
        prefix_error: str = PREFIX_ERROR,
        **kwargs,
    ):

        with_log = (fpath_log is not None)
        if with_log:
            fpath_log.parent.mkdir(parents=True, exist_ok=overwrite)

        with fpath_log.open('w') if with_log else nullcontext() as file_log:
            helper = ScriptHelper(
                file_log=file_log,
                verbosity=verbosity,
                quiet=quiet,
                dry_run=dry_run,
                overwrite=overwrite,
                prefix_run=prefix_run,
                prefix_error=prefix_error,
            )
            try:
                func(helper=helper, **kwargs)
            except Exception:
                helper.print_error_and_exit(traceback.format_exc())

    return _with_helper

def check_dbm_inputs(func):
    @wraps(func)
    def _check_dbm_inputs(
        helper: ScriptHelper,
        dpath_share: Union[None, Path] = None,
        dpath_templates: Union[None, Path] = None,
        template_prefix: str = DEFAULT_TEMPLATE,
        dpath_beast_lib: str = DNAME_BEAST_LIB,
        beast_conf: str = DEFAULT_BEAST_CONF,
        save_all=False,
        **kwargs,
    ):

        # make sure necessary paths are given
        if dpath_share is None and (dpath_templates is None or dpath_beast_lib is None):
                helper.print_error_and_exit('If --share-dir is not given, both '
                                            '--template-dir and --beast-lib-dir '
                                            'must be specified.')
        if dpath_templates is None:
            dpath_templates = dpath_share / DNAME_TEMPLATE_MAP[template_prefix]
        if dpath_beast_lib is None:
            dpath_beast_lib = dpath_share / DNAME_BEAST_LIB

        # generate paths for template files and make sure they are valid
        fpath_template = dpath_templates / f'{template_prefix}{EXT_MINC}'
        fpath_template_mask = add_suffix(fpath_template, 
                                        SUFFIX_TEMPLATE_MASK, sep=None)
        if not fpath_template.exists():
            helper.print_error_and_exit(f'Template file not found: {fpath_template}')
        if not fpath_template_mask.exists():
            helper.print_error_and_exit(
                f'Template mask file not found: {fpath_template_mask}'
            )

        # make sure beast library and config file can be found
        if not dpath_beast_lib.exists():
            helper.print_error_and_exit(
                f'BEaST library directory not found: {dpath_beast_lib}'
            )
        fpath_conf = dpath_beast_lib / beast_conf
        if not fpath_conf.exists():
            helper.print_error_and_exit(f'mincbeast config file not found: {fpath_conf}')

        func(
            helper=helper,
            dpath_templates=dpath_templates,
            template_prefix=template_prefix,
            fpath_template=fpath_template,
            fpath_template_mask=fpath_template_mask,
            dpath_beast_lib=dpath_beast_lib,
            fpath_conf=fpath_conf,
            save_all=save_all,
            **kwargs,
        )

    return _check_dbm_inputs

class ScriptHelper():

    def __init__(
            self,
            file_log=None,
            verbosity=2,
            quiet=False,
            dry_run=False,
            overwrite=False,
            prefix_run=PREFIX_RUN,
            prefix_error=PREFIX_ERROR,
        ) -> None:

        # quiet overrides verbosity
        if quiet:
            verbosity = 0

        self.file_log = file_log
        self.verbosity = verbosity
        self.quiet = quiet
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.prefix_run = prefix_run
        self.prefix_error = prefix_error
    
    def echo(self, message, prefix='', text_color=None, color_prefix_only=False):
        """
        Print a message and newline to stdout or a file, similar to click.echo() 
        but with some color processing.

        Parameters
        ----------
        message : str
            Message to print
        prefix : str, optional
            Prefix to prepend to message, by default ''
        text_color : str or None, optional
            Color name, by default None
        color_prefix_only : bool, optional
            Whether to only color the prefix instead of the entire text, by default False
        """

        # format text to print
        if (prefix != '') and (color_prefix_only):
            text = f'{click.style(prefix, fg=text_color)}{message}'
        else:
            text = click.style(f'{prefix}{message}', fg=text_color)

        click.echo(text, color=True, file=self.file_log)

    def print_error_and_exit(self, message, text_color='red', exit_code=1):
        """Print a message and exit the program.

        Parameters
        ----------
        message : str
            Error message
        text_color : str, optional
            Color name, by default 'red'
        exit_code : int, optional
            Program return code, by default 1
        """
        self.echo(message, prefix=self.prefix_error, text_color=text_color)
        sys.exit(exit_code)

    def run_command(
            self,
            args: list[str],
            shell=False,
            stdout=None,
            stderr=None,
            silent=False,
        ):
        """Run a shell command.

        Parameters
        ----------
        args : list[str]
            Command to pass to subprocess.run()
        shell : bool, optional
            Whether to execute command through the shell, by default False
        stdout : file object, int, or None, optional
            Standard output for executed program, by default None
        stderr : file object, int, or None, optional
            Standard error for execute program, by default None
        silent : bool, optional
            Whether to execute the command without printing the command or the output
        """
        args = [str(arg) for arg in args if arg != '']
        args_str = ' '.join(args)
        if not silent and ((self.verbosity > 0) or self.dry_run):
            self.echo(f'{args_str}', prefix=PREFIX_RUN, text_color='yellow',
                      color_prefix_only=self.dry_run)
        if not self.dry_run:
            if stdout is None:
                if silent or self.verbosity < 2:
                    # note: this doesn't silence everything because some MINC
                    #       tools print a lot of output to stderr
                    stdout = subprocess.DEVNULL
                else:
                    stdout = self.file_log
            if stderr is None:
                stderr = self.file_log
            try:
                subprocess.run(args, check=True, shell=shell,
                               stdout=stdout, stderr=stderr)
            except subprocess.CalledProcessError as ex:
                self.print_error_and_exit(
                    f'Command {args_str} returned {ex.returncode}',
                    exit_code=ex.returncode,
                )

    def timestamp(self):
        """Print the current time."""
        self.run_command(['date'])

    def check_nonempty(self, dpath: Path):
        try:
            if not dpath.exists():
                dpath.mkdir(parents=True)
            if len(list(dpath.iterdir())) != 0 and not self.overwrite:
                raise FileExistsError
        except FileExistsError:
            self.print_error_and_exit(
                f'Output directory {dpath} exists. '
                'Use --overwrite to overwrite'
            )
        return dpath
