#!/usr/bin/env python
import subprocess
import sys

import click

PREFIX_RUN = '[RUN] '
PREFIX_ERROR = '[ERROR] '

class scriptHelper():

    def __init__(
            self,
            file_log=None,
            verbosity=2,
            dry_run=False,
            prefix_run=PREFIX_RUN,
            prefix_error=PREFIX_ERROR,
        ) -> None:

        self.file_log = file_log
        self.verbosity = verbosity
        self.dry_run = dry_run
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

    def run_command(self, args: list[str], shell=False, stdout=None, stderr=None):
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
        """
        args = [str(arg) for arg in args if arg != '']
        args_str = ' '.join(args)
        if (self.verbosity > 0) or self.dry_run:
            self.echo(f'{args_str}', prefix=PREFIX_RUN, text_color='yellow',
                      color_prefix_only=self.dry_run)
        if not self.dry_run:
            if stdout is None:
                if self.verbosity < 2:
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
        """Print the current time.
        """
        self.run_command(['date'])
