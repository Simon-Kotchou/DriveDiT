#!/usr/bin/env python3
"""Measure the bake-off codecs on real harvest derivatives (DD-19), on the local GPU.

Per codec it reports:

- the latent shape, the latent grid positions and the DiT tokens per frame at the patch size planned for that arm;
- encode and decode milliseconds per frame, at the batch size given and at batch 1 (CUDA-synchronised wall clock
  over all frames, after one warm-up batch);
- peak VRAM (``torch.cuda.max_memory_allocated``) from loading through decoding;
- reconstruction PSNR (dB) against the derivative JPEG, per frame, then the mean and the minimum. Diagnostic only:
  codecs are selected by the world model trained on them, never by reconstruction (MODELING_PLAN P2);
- for CADRE, the 5 Hz decode check (MODELING_PLAN §3 and §9): its decoder was trained on 20 fps clips. Each latent is
  decoded as its own one-frame clip (``decode``, what the Codec interface does), and also, in windows of 16 consecutive
  5 Hz latents, as one causal clip (``decode_clip``), so the decoder's temporal layers see 3.2 s of history with about
  4x the motion per step of its training data. Both are scored against the original, and side-by-side PNGs
  (original | one-frame decode | clip decode) are saved for a visual check.

Every codec also gets a few side-by-side PNGs (original | reconstruction) in ``<out>/<codec_id>/``. The results go to
``<out>/measurements.json`` and ``<out>/measurements.md``.

Run it in an environment with CUDA torch and the ``codec`` extra (``uv sync --extra model --extra cuda --extra codec``),
with HF_HOME on the data drive. The CADRE codecs need the verified DINOv3 checkout of ``drivedit.codec.dinov3``
(``DRIVEDIT_DINOV3_DIR`` or its default directory); it is checked first and never fetched::

    python -I -B tools/codec/measure_codecs.py --deriv <session>/_harvest/deriv --out <reports>/codec \\
        --frames 128 --start 1000 --batch 8 --dtype bfloat16

The derivative directory is only read. Weights download on first use unless ``--local-only``; code never downloads.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from drivedit.codec import CODECS, codec_ids, get_codec
from drivedit.codec.base import fetch_weights
from drivedit.codec.dinov3 import dinov3_dir, verify_checkout

# The DiT patch size planned for each arm, so that every arm feeds the probe model about 144 tokens per frame
# (66 for CADRE-1x48x48): delta-2026-11 §2, candidate table and recommendation.
PLANNED_PATCH = {"cadre-1x16x16": 2, "cadre-1x48x48": 1, "flux2-vae": 2, "wan21-vae": 4}
DECISION_HZ = 5  # features_v1 offsets.decision_hz
CLIP = 16  # CADRE's decoder trains on 16-frame clips (checkpoint config num_frames)
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def read_frames(deriv: Path, start: int, count: int) -> torch.Tensor:
    """``[count, 3, 288, 512]`` uint8 RGB of the derivatives ``f_{start..start+count-1:08d}.jpg``."""
    out = []
    for slot in range(start, start + count):
        path = deriv / f"f_{slot:08d}.jpg"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"cannot read {path}")
        out.append(np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1)))
    return torch.from_numpy(np.stack(out))


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-frame PSNR in dB of two ``[N, 3, H, W]`` uint8 batches."""
    mse = (a.double() - b.double()).square().mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return 10.0 * torch.log10(255.0**2 / mse)


def timed(fn, inputs: torch.Tensor, batch: int) -> tuple[torch.Tensor, float]:
    """``fn`` over ``inputs`` in batches; returns the concatenated output and the ms per item (after a warm-up)."""
    fn(inputs[:batch])
    torch.cuda.synchronize()
    begin = time.perf_counter()
    outputs = [fn(inputs[i : i + batch]) for i in range(0, inputs.shape[0], batch)]
    torch.cuda.synchronize()
    return torch.cat(outputs), (time.perf_counter() - begin) * 1000.0 / inputs.shape[0]


def save_side_by_side(path: Path, *images: torch.Tensor) -> None:
    """``[3, H, W]`` uint8 RGB images side by side with a 4-pixel white gap, as a PNG."""
    gap = torch.full((3, images[0].shape[1], 4), 255, dtype=torch.uint8)
    parts = [part for image in images for part in (image.cpu(), gap)][:-1]
    rgb = torch.cat(parts, dim=2).permute(1, 2, 0).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(rgb[:, :, ::-1])):
        raise SystemExit(f"cannot write {path}")


def measure(codec_id: str, images: torch.Tensor, args: argparse.Namespace) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    codec = get_codec(codec_id, device="cuda", dtype=DTYPES[args.dtype])
    load_s = time.perf_counter() - begin
    weights_gib = torch.cuda.memory_allocated() / 2**30
    gpu = images.to(codec.device)
    c, h, w = codec.latent_shape(*images.shape[2:])
    latents, encode_ms = timed(codec.encode, gpu, args.batch)
    decoded, decode_ms = timed(codec.decode, latents, args.batch)
    _, encode_ms_1 = timed(codec.encode, gpu[: args.batch * 2], 1)
    _, decode_ms_1 = timed(codec.decode, latents[: args.batch * 2], 1)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    scores = psnr(decoded, gpu)
    out_dir = args.out / codec_id
    picks = sorted({0, images.shape[0] // 2, images.shape[0] - 1})
    for i in picks:
        save_side_by_side(out_dir / f"recon_{args.start + i:08d}.png", gpu[i], decoded[i])
    patch = PLANNED_PATCH[codec_id]
    result = {
        "codec_id": codec_id,
        "weights": f"{codec.weights.repo_id}@{codec.weights.revision[:8]}",
        "licence": codec.weights.licence,
        "latent_shape": [c, h, w],
        "grid_positions": h * w,
        "dit_patch": patch,
        "tokens_per_frame": math.ceil(h / patch) * math.ceil(w / patch),
        "latent_kib_per_frame_fp16": c * h * w * 2 / 1024,
        "latent_gb_per_hour_fp16": c * h * w * 2 * DECISION_HZ * 3600 / 1e9,
        "load_s": load_s,
        "weights_gib": weights_gib,
        "encode_ms_per_frame": encode_ms,
        "decode_ms_per_frame": decode_ms,
        "encode_ms_batch1": encode_ms_1,
        "decode_ms_batch1": decode_ms_1,
        "peak_vram_gib": peak_gib,
        "psnr_db_mean": scores.mean().item(),
        "psnr_db_min": scores.min().item(),
        "latent_mean": latents.mean().item(),
        "latent_std": latents.std().item(),
        "pngs": [str(out_dir / f"recon_{args.start + i:08d}.png") for i in picks],
    }
    if CODECS[codec_id].module == "cadre":
        result["cadre"] = measure_cadre(codec, gpu, latents, decoded, scores, args)
    del codec, gpu, latents, decoded
    gc.collect()
    torch.cuda.empty_cache()
    return result


def measure_cadre(codec, gpu, latents, decoded, scores, args) -> dict:
    """The 5 Hz decode check: one-frame decodes against 16-frame causal clip decodes, plus 2-step sampling."""
    windows = range(0, latents.shape[0] - CLIP + 1, CLIP)
    clip = torch.cat([codec.decode_clip(latents[s : s + CLIP]) for s in windows])
    covered = len(windows) * CLIP
    clip_scores = psnr(clip, gpu[:covered])
    position = clip_scores.view(-1, CLIP).mean(dim=0)  # mean PSNR by position in the clip (history length)
    out_dir = args.out / codec.codec_id
    pngs = []
    for s in list(windows)[:2]:
        for offset in (0, CLIP // 2, CLIP - 1):
            i = s + offset
            path = out_dir / f"5hz_{args.start + i:08d}_pos{offset:02d}.png"
            save_side_by_side(path, gpu[i], decoded[i], clip[i])
            pngs.append(str(path))
    codec.decode_steps = 2
    two_step = torch.cat([codec.decode(latents[i : i + args.batch]) for i in range(0, latents.shape[0], args.batch)])
    codec.decode_steps = 1
    two_scores = psnr(two_step, gpu)
    return {
        "frames_in_clips": covered,
        "psnr_db_one_frame_decode": scores[:covered].mean().item(),
        "psnr_db_clip_decode": clip_scores.mean().item(),
        "psnr_db_clip_by_position": [round(v, 2) for v in position.tolist()],
        "psnr_db_two_step_one_frame_decode": two_scores.mean().item(),
        "pngs": pngs,
    }


def markdown(results: list[dict], header: dict) -> str:
    lines = [
        f"# Codec measurements ({header['date']})",
        "",
        f"{header['frames']} frames {header['first']}..{header['last']} of `{header['session']}`; "
        f"{header['gpu']}, torch {header['torch']}, compute dtype {header['dtype']}, batch {header['batch']}. "
        "PSNR is diagnostic only (MODELING_PLAN P2).",
        "",
        "| codec | latent (C, h, w) | tokens/frame (patch) | encode ms/frame (b={b}, b=1) | decode ms/frame (b={b}, b=1) "
        "| peak VRAM GiB | PSNR dB mean / min | latents GB/h fp16 |".format(b=header["batch"]),
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['codec_id']} | {tuple(r['latent_shape'])} | {r['tokens_per_frame']} (p{r['dit_patch']}) "
            f"| {r['encode_ms_per_frame']:.1f}, {r['encode_ms_batch1']:.1f} "
            f"| {r['decode_ms_per_frame']:.1f}, {r['decode_ms_batch1']:.1f} | {r['peak_vram_gib']:.2f} "
            f"| {r['psnr_db_mean']:.2f} / {r['psnr_db_min']:.2f} | {r['latent_gb_per_hour_fp16']:.2f} |"
        )
    cadre = [r for r in results if "cadre" in r]
    if cadre:
        lines += [
            "",
            "CADRE 5 Hz decode check (PSNR dB on the frames covered by 16-latent clips):",
            "",
            "| codec | one-frame decode, 1 step | one-frame decode, 2 steps | 16-frame clip decode, 1 step "
            "| clip decode by position 0 / 7 / 15 |",
            "|---|---|---|---|---|",
        ]
        for r in cadre:
            k = r["cadre"]
            pos = k["psnr_db_clip_by_position"]
            lines.append(
                f"| {r['codec_id']} | {k['psnr_db_one_frame_decode']:.2f} "
                f"| {k['psnr_db_two_step_one_frame_decode']:.2f} | {k['psnr_db_clip_decode']:.2f} "
                f"| {pos[0]:.2f} / {pos[CLIP // 2 - 1]:.2f} / {pos[-1]:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--deriv", type=Path, required=True, help="a session's _harvest/deriv directory (read only)")
    parser.add_argument("--out", type=Path, required=True, help="report directory (created)")
    parser.add_argument("--codecs", nargs="+", choices=codec_ids(), default=list(codec_ids()))
    parser.add_argument("--start", type=int, default=0, help="first derivative slot")
    parser.add_argument("--frames", type=int, default=128, help="consecutive frames, 64-256 (default 128)")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--local-only", action="store_true", help="fail instead of downloading missing weights")
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    if args.frames < CLIP:
        raise SystemExit(f"--frames must be at least {CLIP}")
    if any(CODECS[codec_id].module == "cadre" for codec_id in args.codecs):
        verify_checkout(dinov3_dir())  # DinoV3SourceError, naming the fetch command, before any work
    if args.local_only:
        for codec_id in args.codecs:
            fetch_weights(CODECS[codec_id].weights, local_only=True)
    images = read_frames(args.deriv, args.start, args.frames)
    args.out.mkdir(parents=True, exist_ok=True)
    header = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "session": args.deriv.parent.parent.name,
        "first": args.start,
        "last": args.start + args.frames - 1,
        "frames": args.frames,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": args.dtype,
        "batch": args.batch,
    }
    results = []
    for codec_id in args.codecs:
        print(f"measuring {codec_id} ...", file=sys.stderr, flush=True)
        results.append(measure(codec_id, images, args))
        print(json.dumps({k: v for k, v in results[-1].items() if k != "pngs"}), file=sys.stderr, flush=True)
    (args.out / "measurements.json").write_text(json.dumps({"header": header, "results": results}, indent=2) + "\n")
    table = markdown(results, header)
    (args.out / "measurements.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
