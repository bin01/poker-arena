"""Temporary folders for every test: the system's (TMPDIR first) when it is writable, else tests/.tmp (kept in the
repository, so it exists even where nothing else may be created), so the suite also runs where the system temp
folder is read-only or TMPDIR is unset or unwritable. The choice is made when this file is imported, before any
test or pytest's tmp_path asks for a folder; subprocesses inherit it through TMPDIR. pytest's own capture needs
no temporary file (`--capture=sys` in pyproject.toml), so nothing before this file needs one."""
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _writable(d) -> str | None:
    """None when `d` can hold temporary files, else why not."""
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=d):
            pass
        tempfile.TemporaryDirectory(dir=d).cleanup()
        return None
    except OSError as e:
        return f'{d}: {e.strerror or e}'


def _choose() -> str:
    tried = []
    for d in (os.environ.get('TMPDIR'), os.environ.get('TEMP'), os.environ.get('TMP')):
        if d:
            why = _writable(d)
            if not why:
                return str(Path(d).resolve())
            tried.append(why)
    for d in ('/tmp', '/var/tmp'):
        why = _writable(d)
        if not why:
            return d
        tried.append(why)
    d = HERE / '.tmp'
    why = _writable(d)
    if why:
        raise RuntimeError('no writable temporary folder; tried ' + '; '.join(tried + [why]) +
                           f'. Make {d} writable, or set TMPDIR to a writable folder')
    return str(d)


tempfile.tempdir = _choose()
os.environ['TMPDIR'] = tempfile.tempdir
