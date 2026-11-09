"""Opt-in machine evidence for the tests (DD-22).

A few tests compare repo files with documents or data of a real machine (the capture program, the real legacy and
captured sessions, the official script dump) or run on a real drvfs scratch directory. They run only when
DD_TEST_LOCAL_CONFIG names that machine's drivedit.local.env; everywhere else, CI included, they skip. Every other
test reads the synthetic config placeholder.local.env next to this file and names no real path.

    spec = importlib.util.spec_from_file_location("dd_test_evidence", "<repo>/tests/fixtures/local_config/evidence.py")
"""

import importlib.util
import os

ENV = "DD_TEST_LOCAL_CONFIG"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FIXTURE = os.path.join(HERE, "placeholder.local.env")
LOADER = os.path.join(REPO, "tools", "local_config.py")


def _loader():
    spec = importlib.util.spec_from_file_location("dd_test_evidence_local_config", LOADER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def machine_config_path():
    """The file DD_TEST_LOCAL_CONFIG names, or None (the opt-in is off)."""
    path = os.environ.get(ENV, "").strip()
    return path or None


def evidence_path(key, *rel):
    """<KEY of the opted-in machine config>/<rel...>, or None when the opt-in is off or the key is missing."""
    path = machine_config_path()
    if not path:
        return None
    lc = _loader()
    try:
        return os.path.join(lc.load(path).get(key), *rel)
    except lc.LocalConfigError:
        return None


def skip_reason(what):
    return "set %s to this machine's drivedit.local.env to compare with %s (DD-22 opt-in)" % (ENV, what)
