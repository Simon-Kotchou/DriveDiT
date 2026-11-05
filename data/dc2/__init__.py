"""dc-2.0 Python tools: fiducial decoder (DD-4-probe), reader (DD-5-min), reference crash detector (crash_dv).

This package imports nothing on purpose. The probe tools, the validator, the harvester and the tests load its modules
by file path (importlib.util.spec_from_file_location). The parent package `data` imports nothing either (DD-16), so
`import data.dc2.reader` with the repo root on sys.path works too and needs no torch.
"""
