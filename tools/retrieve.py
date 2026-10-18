"""Search open-licensed reference photos of a PUBLIC place (algorithm A-4). Metadata only: lists
candidates with licence and attribution; it downloads no images.

    python -m tools.retrieve search "Vasai Fort"

Needs `[retrieval] contact` in worldreel.local.toml (Wikimedia User-Agent policy). Never use it
with a home address or a private place (brief §6.3).
"""

from __future__ import annotations

import argparse

from stages.fill.retrieve import (
    Candidate,
    commons_search,
    local_contact,
    mentions_people,
    openverse_search,
)


def search(query: str) -> list[Candidate]:
    contact = local_contact()
    found = openverse_search(query, contact) + commons_search(query, contact)
    seen: set[str] = set()
    out = []
    for c in found:
        if c.url and c.url not in seen:
            seen.add(c.url)
            out.append(c)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.retrieve")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search", help="list licence-compatible candidates for a public place")
    s.add_argument("query")
    a = p.parse_args(argv)
    cands = search(a.query)
    usable = [c for c in cands if not mentions_people(c.metadata_text)]
    print(f"{len(cands)} licence-compatible candidates, {len(usable)} without people words")
    for c in usable:
        title = " ".join(c.metadata_text.split())[:70]
        print(f"- [{c.source}] {c.license:14s} {title}")
        print(f"    {c.landing_url or c.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
