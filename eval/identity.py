"""A-8 measurement: how often is a cast too similar, with and without best-candidate sampling?

    python -m eval.identity [--casts 300]

For cast sizes 2-40 with no style requests (every dimension free), the share of casts that fail
the distinctness thresholds (`bible.APPEARANCE_MIN`, `bible.VOICE_MIN`) for: independent seeds
(baseline, K = 1), best of 8 and best of 32 candidates. Plus a hard case — every character
requested as "older bearded man, deep voice" — and the collapse counter on a collapsed batch.
"""

from __future__ import annotations

import argparse

from stages.generate.bible import check_distinct, expand, sample_cast, unique_identities

SIZES = (2, 4, 8, 12, 20, 40)
KS = (1, 8, 32)


def failure_rates(casts: int, requests_for: str = "") -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for n in SIZES:
        req = {f"C{i:02d}": requests_for for i in range(n)}
        out[n] = {}
        for k in KS:
            fails = sum(not check_distinct(sample_cast(req, seed, k)).ok for seed in range(casts))
            out[n][k] = fails / casts
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.identity")
    p.add_argument("--casts", type=int, default=300)
    a = p.parse_args(argv)
    print("free:", failure_rates(a.casts))
    print("hard:", failure_rates(max(a.casts // 3, 10), "older bearded man, deep voice"))
    collapsed = [expand(7) for _ in range(20)]
    varied = list(sample_cast({f"C{i:02d}": "" for i in range(20)}, 0).values())
    print(
        "collapse counter: one seed x20 ->",
        unique_identities(collapsed),
        "| sampled cast of 20 ->",
        unique_identities(varied),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
