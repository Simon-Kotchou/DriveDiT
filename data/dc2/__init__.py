"""dc-2.0 Python tools: fiducial decoder (DD-4-probe), reader (DD-5-min).

This package imports nothing on purpose. The parent package `data` imports torch and the dc-1.0 loaders, so the
probe tools and tests load `data/dc2/fiducial.py` by file path (importlib.util.spec_from_file_location) instead of
`import data.dc2.fiducial`.
"""
