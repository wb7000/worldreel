"""Kaggle jobs ship their real program as an embedded script.

A nested triple-quote inside that script truncates it, so the job parses
locally and then dies on the GPU after the queue wait. This guards both
directions: every real job must pass, and a deliberately broken job must fail
(a checker that cannot fail is not a checker).
"""

from __future__ import annotations

from pathlib import Path

from tools.job_script_check import check

JOBS = sorted(Path("training/kaggle/jobs").glob("*.py"))


def test_every_job_script_parses() -> None:
    assert JOBS, "no Kaggle jobs found"
    failures = check(JOBS)
    assert not failures, "\n".join(failures)


def test_broken_outer_file_is_caught(tmp_path: Path) -> None:
    # A nested triple-quote breaks the job file itself -- the failure mode that
    # hit character_consistency_v2.py (D-9.13 prep).
    bad = tmp_path / "bad_outer.py"
    bad.write_text(
        "import textwrap\n"
        "step.write_text(\n"
        '    textwrap.dedent("""\n'
        "    def build():\n"
        '        """Docstring."""\n'
        "        return 1\n"
        '    """),\n'
        '    encoding="utf-8",\n'
        ")\n",
        encoding="utf-8",
    )
    failures = check([bad])
    assert failures, "a job file with a nested triple-quote must not pass"


def test_broken_embedded_script_is_caught(tmp_path: Path) -> None:
    # The job file is valid Python but the source it ships is not: only the
    # AST extraction path can see this one.
    bad = tmp_path / "bad_inner.py"
    bad.write_text(
        "import textwrap\n"
        'code = "def build(:\\n    return 1\\n"\n'
        'step.write_text(textwrap.dedent(code), encoding="utf-8")\n',
        encoding="utf-8",
    )
    failures = check([bad])
    assert failures, "an embedded script with a syntax error must not pass"
    assert "embedded script" in failures[0]


def test_one_line_write_text_is_not_mistaken_for_a_script(tmp_path: Path) -> None:
    # marker.write_text("...\\n") is data, not code; flagging it would be a
    # bogus failure that trains people to ignore this check.
    good = tmp_path / "data_write.py"
    good.write_text(
        "marker = out / 'm.txt'\n"
        'marker.write_text("worldreel output path test\\n", encoding="utf-8")\n',
        encoding="utf-8",
    )
    assert check([good]) == []
