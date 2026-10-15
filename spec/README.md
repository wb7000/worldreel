# Format specifications

Open, versioned specs. Written **before** any implementation. Phase 1 adds:

- `wreel.md` — interactive world container (`.wreel`)
- `wedit.md` — edit bundle (`.wedit`)
- `wscript.md` — long-script format (`.wscript`)

Rules: every file carries `format_version`; readers must keep older versions readable; breaking
changes bump the major version and are logged in `DECISIONS.md`.
