import os
from pathlib import Path

_ENV_VAR = "MOLFUSION_ARTIFACT_ROOT"

# backend/src/molfusion_backend/artifacts/root.py -> backend/
_BACKEND_DIR = Path(__file__).resolve().parents[3]


def default_artifact_root() -> Path:
    """`<backend>/artifacts`, derived from this file's own location.

    Deliberately not based on the current working directory: the API can be
    launched from any directory (e.g. `uv run uvicorn ...` from a different
    cwd), and the artifact root must not move when that changes.
    """
    return _BACKEND_DIR / "artifacts"


def resolve_artifact_root() -> Path:
    """The effective artifact root for this process.

    Precedence: the `MOLFUSION_ARTIFACT_ROOT` environment variable, if set
    to a non-empty value, overrides `default_artifact_root()`. No env var
    is required for normal development -- the default just works.
    """
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).resolve()
    return default_artifact_root()


def validate_path_component(value: str, field_name: str) -> str:
    """Reject a string that is unsafe to use as a single path segment.

    Used for `artifact_type`, `artifact_id`, and `artifact_version` values,
    which are joined onto the artifact root to build a directory path.
    Rejects the empty string, `.`/`..`, any path separator, and leading/
    trailing whitespace -- all of which could otherwise be used to escape
    the artifact root or reinterpret the path (e.g. "../../secret").
    """
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty, non-whitespace-padded string")
    if value in (".", ".."):
        raise ValueError(f"{field_name} must not be '.' or '..'")
    if "/" in value or "\\" in value or os.sep in value or (os.altsep and os.altsep in value):
        raise ValueError(f"{field_name} must not contain a path separator: {value!r}")
    return value
