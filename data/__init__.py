"""DriveDiT data code: dc-2.0 sessions are read by data/dc2/reader.py (docs/capture_contract/dc-2.0.md).

This package imports nothing, so importing it (or data.dc2) never pulls in torch. The pre-dc-2.0 pipeline that lived
here (video loader, transforms, collate, samplers, preprocessing, chunking, large-scale processing and the dc-1.0
telemetry preprocessing) was removed in DD-16 and is kept in tag archive/model-2026-03. The dc-1.0 session loader, the
Rust loaders and the streaming loader were removed in DD-12 and are kept in tag archive/dc1-rust.
"""
