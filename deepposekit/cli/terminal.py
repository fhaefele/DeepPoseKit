"""Terminal presentation for the command line."""

import argparse
import os
import re
import sys

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich_argparse import RichHelpFormatter

COLORS = ("auto", "always", "never")


class HelpFormatter(RichHelpFormatter):
    styles = {
        **RichHelpFormatter.styles,
        "argparse.groups": "bold yellow",
        "argparse.args": "bold cyan",
        "argparse.metavar": "cyan",
        "argparse.prog": "bold cyan",
    }
    text_markup = False
    help_markup = False


def color_mode(argv):
    selected = "auto"
    for index, token in enumerate(argv):
        if token == "--":
            break
        if token.startswith("--color="):
            value = token.partition("=")[2]
        elif token == "--color" and index + 1 < len(argv):
            value = argv[index + 1]
        else:
            continue
        if value in COLORS:
            selected = value
    return selected


def colors_enabled(stream, color):
    return color == "always" or (
        color == "auto"
        and stream.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )


class TrainingProgress:
    """Apply CLI colors and keep subsequent diagnostics on their own line."""

    _ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

    def __init__(self, stream, colored):
        self.stream = stream
        self.colored = colored
        self.line_complete = True

    def write(self, text):
        styled = text if self.colored else self._ansi.sub("", text)
        if styled:
            self.stream.write(styled)
            self.line_complete = styled.endswith("\n")
        return len(text)

    def finish(self):
        if not self.line_complete:
            self.stream.write("\n")

    def __getattr__(self, name):
        return getattr(self.stream, name)


def console_for(stream, color):
    tty = stream.isatty()
    width = 100
    if tty:
        try:
            width = os.get_terminal_size(stream.fileno()).columns or width
        except (AttributeError, OSError, ValueError):
            pass
    enabled = colors_enabled(stream, color)
    return Console(
        file=stream,
        width=width,
        force_terminal=enabled,
        color_system="standard" if enabled else None,
        no_color=not enabled,
        highlight=False,
    )


class TerminalParser(argparse.ArgumentParser):
    def __init__(self, *args, color="auto", **kwargs):
        self.color = color
        self.stream = None
        self.command_groups = {}
        self.reference_notes = []
        super().__init__(
            *args, formatter_class=HelpFormatter, allow_abbrev=False, **kwargs
        )
        self.add_argument(
            "--color",
            choices=COLORS,
            default=argparse.SUPPRESS,
            help="Terminal colors (default: auto).",
        )

    def _get_formatter(self):
        console = console_for(self.stream or sys.stdout, self.color)
        formatter = self.formatter_class(
            prog=self.prog,
            width=console.width,
            max_help_position=min(console.width // 2, 24),
        )
        formatter.console = console
        return formatter

    def format_help(self):
        if not self.command_groups:
            return super().format_help()
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        formatter.start_section("Options")
        formatter.add_arguments(
            [action for action in self._actions if action.option_strings]
        )
        formatter.end_section()
        console = formatter.console
        with console.capture() as output:
            for heading, entries in self.command_groups.items():
                console.print(Text(f"\n{heading}", style="bold yellow"))
                if console.width >= 60:
                    table = Table.grid(padding=(0, 2))
                    table.add_column(style="bold cyan", no_wrap=True)
                    table.add_column(overflow="fold")
                    for name, description in entries:
                        table.add_row(Text(f"  {name}"), Text(description))
                    console.print(table)
                else:
                    for name, description in entries:
                        console.print(Text(f"  {name}", style="bold cyan"))
                        console.print(Text(f"    {description}"), overflow="fold")
        return formatter.format_help() + output.get()

    def print_help(self, file=None):
        self._print_formatted(self.format_help, file or sys.stdout)

    def print_usage(self, file=None):
        self._print_formatted(self.format_usage, file or sys.stdout)

    def _print_formatted(self, format_text, stream):
        self.stream = stream
        try:
            console = console_for(stream, self.color)
            console.print(Text.from_ansi(format_text()), end="", overflow="fold")
        finally:
            self.stream = None

    def error(self, message):
        self.print_usage(sys.stderr)
        self.show_error(message)
        self.exit(2)

    def show_error(self, message):
        console_for(sys.stderr, self.color).print(
            Text.assemble((f"{self.prog}: error: ", "red"), str(message)),
            overflow="fold",
        )
