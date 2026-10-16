from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from formats.__main__ import main
from formats.wreel import write_wreel


def test_cli_validate(
    tmp_path: Path,
    sample: tuple[dict[str, Any], dict[str, bytes]],
    key: Ed25519PrivateKey,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = tmp_path / "good.wreel"
    write_wreel(good, sample[0], sample[1], key)
    bad = tmp_path / "bad.wreel"
    bad.write_bytes(b"nope")
    assert main(["validate", str(good)]) == 0
    assert main(["validate", str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert "good.wreel: OK" in out
    assert "bad.wreel: INVALID" in out
