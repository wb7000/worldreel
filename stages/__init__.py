"""WorldReel pipeline stages. One subpackage per stage; stages communicate only through files."""

STAGES: tuple[str, ...] = (
    "ingest",
    "frames",
    "camera",
    "reconstruct",
    "actors",
    "fill",
    "audio",
    "package",
    "export",
    "generate",
    "safety",
)
