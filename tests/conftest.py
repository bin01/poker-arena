"""Temporary folders for every test: the system's (TMPDIR first) when it is writable, else tests/.tmp, so the
suite also runs where the system temp folder is read-only. Subprocesses inherit the choice through TMPDIR."""
import os
import tempfile
from pathlib import Path


def _writable(d) -> bool:
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        tempfile.TemporaryDirectory(dir=d).cleanup()
        return True
    except OSError:
        return False


def _choose() -> str:
    for d in (os.environ.get('TMPDIR'), os.environ.get('TEMP'), os.environ.get('TMP')):
        if d and _writable(d):
            return d
    try:
        d = tempfile.gettempdir()
        if Path(d).resolve() != Path.cwd().resolve() and _writable(d):
            return d
    except (OSError, FileNotFoundError):
        pass
    d = Path(__file__).resolve().parent / '.tmp'
    if not _writable(d):
        raise RuntimeError(f'no writable temporary folder: set TMPDIR, or make {d} writable')
    return str(d)


tempfile.tempdir = _choose()
os.environ['TMPDIR'] = tempfile.tempdir
