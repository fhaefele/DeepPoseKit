"""Tests for the versioned CLI reference."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import unittest

from deepposekit import __version__
from deepposekit.cli import main


def invoke(*args, **kwargs):
    stdout = StringIO()
    with redirect_stdout(stdout):
        status = main(list(args), **kwargs)
    if status:
        raise AssertionError(status)
    return stdout.getvalue()


class ReferenceTests(unittest.TestCase):
    def test_json_is_colorless_and_exposes_constraints(self):
        output = invoke(
            "--color", "always", "help", "--format", "json", "dataset", "split"
        )
        self.assertNotIn("\x1b", output)
        document = json.loads(output)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["deepposekit_version"], __version__)
        command = document["command"]
        self.assertEqual(command["path"], ["dataset", "split"])
        self.assertIn(
            {"required": True, "arguments": ["--frames", "--at"]},
            command["mutually_exclusive"],
        )
        self.assertTrue(any("exclude the stop" in note for note in command["notes"]))
        self.assertTrue(
            all(
                example["argv"][:2] == ["dataset", "split"]
                for example in command["examples"]
            )
        )

    def test_trace_reference_exposes_numeric_confidence_threshold(self):
        document = json.loads(
            invoke("help", "--format", "json", "predictions", "trace")
        )
        threshold = next(
            field
            for field in document["command"]["arguments"]
            if field["name"] == "confidence_threshold"
        )
        self.assertEqual(threshold["type"], "number")

    def test_embedded_reference_uses_its_configured_flags_and_version(self):
        def configure_predict(parser):
            parser.add_argument("--camera", type=int, required=True)

        document = json.loads(
            invoke(
                "help",
                "--format",
                "json",
                "predict",
                prog="wrapper",
                version="42",
                configure_predict=configure_predict,
            )
        )
        self.assertEqual(document["program"], "wrapper")
        self.assertEqual(document["program_version"], "42")
        self.assertEqual(document["deepposekit_version"], __version__)
        fields = {
            flag: item
            for item in document["command"]["arguments"]
            for flag in item["flags"]
        }
        self.assertTrue(fields["--camera"]["required"])
        self.assertNotIn("--custom-reader", fields)
        self.assertEqual(document["command"]["examples"], [])

    def test_wrapped_reader_commands_omit_generic_examples(self):
        def configure_reader(parser, *, role):
            parser.add_argument("--camera", required=True)

        for path in (
            ("dataset", "init"),
            ("dataset", "append"),
            ("predictions", "view"),
        ):
            with self.subTest(path=path):
                document = json.loads(
                    invoke(
                        "help",
                        "--format",
                        "json",
                        *path,
                        configure_reader=configure_reader,
                    )
                )
                command = document["command"]
                camera = next(
                    arg for arg in command["arguments"] if arg["name"] == "camera"
                )
                self.assertTrue(camera["required"])
                self.assertEqual(command["examples"], [])

    def test_help_exit_and_starts_without_tensorflow(self):
        out = StringIO()
        with redirect_stderr(out), self.assertRaises(SystemExit) as error:
            main(["help", "dataset", "unknown"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Unknown command", out.getvalue())
        self.assertIn("--frames", invoke("help", "dataset", "split"))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from deepposekit.cli import main; import sys; "
                "assert main(['help', '--format', 'json']) == 0; "
                "assert 'tensorflow' not in sys.modules",
            ],
            check=True,
            capture_output=True,
        )

    def test_positional_help_path_is_optional(self):
        output = invoke("help", "--format", "json", "help")
        path = next(
            argument
            for argument in json.loads(output)["command"]["arguments"]
            if argument["name"] == "path"
        )
        self.assertEqual(path["nargs"], "*")
        self.assertFalse(path["required"])
        self.assertIn("| `path` | No |", invoke("help", "--format", "markdown", "help"))

    def test_checked_in_markdown_matches_parser(self):
        reference = Path(__file__).resolve().parents[1] / "docs" / "cli.md"
        self.assertEqual(
            reference.read_text(encoding="utf-8"),
            invoke("help", "--format", "markdown"),
        )
