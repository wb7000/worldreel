from __future__ import annotations

import copy
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from formats.common import generate_signing_key
from tools.make_sample_wreel import build_sample

FIXED_TIME = "2026-10-16T00:00:00+00:00"


@pytest.fixture(scope="session")
def key() -> Ed25519PrivateKey:
    return generate_signing_key()


@pytest.fixture(scope="session")
def _sample_cache() -> tuple[dict[str, Any], dict[str, bytes]]:
    return build_sample(seed=7, created_at=FIXED_TIME)


@pytest.fixture
def sample(
    _sample_cache: tuple[dict[str, Any], dict[str, bytes]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """A fresh deep copy of the synthetic sample scene for each test to modify."""
    manifest, entries = _sample_cache
    return copy.deepcopy(manifest), dict(entries)
