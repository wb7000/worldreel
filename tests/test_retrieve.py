from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from stages.fill.retrieve import (
    FACE_MODEL,
    commons_geosearch,
    has_face,
    mentions_people,
    normalise_license,
    openverse_search,
    user_agent,
    user_candidates,
)


@pytest.mark.parametrize(
    ("text", "version", "expected"),
    [
        ("cc0", "", "CC0-1.0"),
        ("pdm", "", "PDM"),
        ("by", "4.0", "CC-BY-4.0"),
        ("by-sa", "2.0", "CC-BY-SA-2.0"),
        ("CC BY-SA 4.0", "", "CC-BY-SA-4.0"),
        ("CC BY 3.0", "", "CC-BY-3.0"),
        ("CC-BY-4.0", "", "CC-BY-4.0"),
        ("CC-BY-SA-2.0", "", "CC-BY-SA-2.0"),
        ("CC0-1.0", "", "CC0-1.0"),
        ("Public domain", "", "PDM"),
        ("CC0", "", "CC0-1.0"),
        ("by-nc", "4.0", None),  # non-commercial: refused
        ("by-nd", "4.0", None),  # no derivatives: refused (reconstruction is a derivative)
        ("CC BY-NC-SA 4.0", "", None),
        ("GFDL", "", None),  # unknown to the allow-list: refused
        ("", "", None),
    ],
)
def test_normalise_license(text: str, version: str, expected: str | None) -> None:
    assert normalise_license(text, version) == expected


def test_people_and_interior_words_are_filtered() -> None:
    assert mentions_people("Portrait of a woman at the market")
    assert mentions_people("Category:Living room interiors")
    assert not mentions_people("Clock tower, north facade, 2019")


def test_network_needs_a_contact() -> None:
    with pytest.raises(PermissionError):
        user_agent("")
    assert "contact@example.org" in user_agent("contact@example.org")


def test_user_folder_needs_allowed_licence(tmp_path: Path) -> None:
    img = np.full((40, 60, 3), 120, np.uint8)
    for name, lic in (("a.png", "CC-BY-4.0"), ("b.png", "CC BY-NC 4.0"), ("c.png", None)):
        cv2.imwrite(str(tmp_path / name), img)
        if lic:
            (tmp_path / f"{name}.info.json").write_text(
                json.dumps({"license": lic}), encoding="utf-8"
            )
    got = user_candidates(tmp_path)
    assert [c.url for c in got] == ["user:a.png"] and got[0].image is not None


# Offline responses shaped like the documented API fields (constructed for the test, not
# recorded from the services): Openverse /v1/images/ and MediaWiki geosearch + imageinfo.
def _fake(payload: dict[str, Any]) -> tuple[Any, list[tuple[str, dict[str, str]]]]:
    seen: list[tuple[str, dict[str, str]]] = []

    def get(url: str, headers: dict[str, str]) -> bytes:
        seen.append((url, headers))
        return json.dumps(payload).encode()

    return get, seen


def test_openverse_parser_filters_licences_and_sends_user_agent() -> None:
    payload = {
        "results": [
            {
                "url": "https://x/a.jpg",
                "license": "by-sa",
                "license_version": "4.0",
                "attribution": "A by B",
                "foreign_landing_url": "https://x/a",
                "title": "Tower",
                "tags": [{"name": "clock"}],
                "mature": False,
            },
            {
                "url": "https://x/b.jpg",
                "license": "by-nc",
                "license_version": "2.0",
                "mature": False,
            },
        ]
    }
    get, seen = _fake(payload)
    got = openverse_search("old town clock tower", "me@example.org", get=get)
    assert [c.license for c in got] == ["CC-BY-SA-4.0"]
    url, headers = seen[0]
    assert (
        url.startswith("https://api.openverse.org/v1/images/?")
        and "license=cc0%2Cpdm%2Cby%2Cby-sa" in url
    )
    assert "me@example.org" in headers["User-Agent"]


def test_commons_parser_skips_multi_licensed_and_nc() -> None:
    def page(title: str, lic: str) -> dict[str, Any]:
        return {
            "title": title,
            "imageinfo": [
                {
                    "url": f"https://u/{title}",
                    "descriptionurl": "d",
                    "extmetadata": {
                        "LicenseShortName": {"value": lic},
                        "Artist": {"value": "<a>X</a>"},
                        "Categories": {"value": "Bridges"},
                    },
                }
            ],
        }

    payload = {
        "query": {
            "pages": {
                "1": page("File:A.jpg", "CC BY 4.0"),
                "2": page("File:B.jpg", "CC BY-NC 2.0"),
                "3": page("File:C.jpg", "CC BY-SA 3.0|GFDL"),
            }
        }
    }
    get, seen = _fake(payload)
    got = commons_geosearch(48.85, 2.29, "me@example.org", get=get)
    assert [c.url for c in got] == ["https://u/File:A.jpg"] and got[0].attribution == "X"
    assert "ggscoord=48.85%7C2.29" in seen[0][0]


@pytest.mark.skipif(not FACE_MODEL.is_file(), reason="YuNet model not downloaded (docs/MODELS.md)")
def test_face_detector_quiet_on_an_empty_scene() -> None:
    sky = np.zeros((240, 320, 3), np.uint8)
    sky[:] = (140, 170, 210)
    assert not has_face(sky)
