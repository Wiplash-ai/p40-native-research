#!/usr/bin/env python3
"""Bounded FP16/SDPA canary for ARC-Qwen Narrator on a Pascal P40.

This is intentionally not a service.  Run it under the existing thermal
supervisor after an explicit cooling preflight, then promote the validated
execution path into the private media gateway.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


MAX_DURATION_SECONDS = 15.0
MAX_NEW_TOKENS = 128


def iso_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Pinned ARC-Hunyuan-Video-7B checkout")
    parser.add_argument("--model", required=True, help="Local ARC-Qwen Narrator checkpoint")
    parser.add_argument("--video", required=True, help="Disposable canary proxy, never canonical media")
    parser.add_argument("--output", required=True, help="JSON result path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION_SECONDS)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.max_duration <= MAX_DURATION_SECONDS:
        raise SystemExit(f"--max-duration must be in (0, {MAX_DURATION_SECONDS}]")
    if not 1 <= args.max_new_tokens <= MAX_NEW_TOKENS:
        raise SystemExit(f"--max-new-tokens must be in [1, {MAX_NEW_TOKENS}]")

    repo = Path(args.repo).resolve()
    model_path = Path(args.model).resolve()
    video_path = Path(args.video).resolve()
    output_path = Path(args.output).resolve()
    if not (repo / "vision_process.py").is_file():
        raise SystemExit(f"missing ARC source checkout: {repo}")
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"missing ARC model checkpoint: {model_path}")
    if not video_path.is_file():
        raise SystemExit(f"missing canary video: {video_path}")

    duration = media_duration(video_path)
    if duration > args.max_duration:
        raise SystemExit(
            f"canary duration {duration:.3f}s exceeds bounded limit {args.max_duration:.3f}s"
        )

    sys.path.insert(0, str(repo))
    import torch
    from transformers import (  # pylint: disable=import-outside-toplevel
        ARC_Qwen2_5_VL_VideoForConditionalGeneration,
        AutoProcessor,
        WhisperFeatureExtractor,
    )
    from vision_process import process_vision_info  # pylint: disable=import-outside-toplevel
    from vision_utils import load_audio_from_video  # pylint: disable=import-outside-toplevel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.cuda.set_device(args.device)
    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability != (6, 1):
        raise SystemExit(f"this P40 canary expects sm_61, got {capability}")

    result: dict[str, Any] = {
        "schema_version": 1,
        "started_at": iso_utc(),
        "completed_at": None,
        "status": "started",
        "precision": "float16",
        "attention": "sdpa",
        "model": str(model_path),
        "video": str(video_path),
        "video_sha256": sha256_file(video_path),
        "duration_seconds": round(duration, 3),
        "max_new_tokens": args.max_new_tokens,
        "device": torch.cuda.get_device_name(device_index),
        "capability": list(capability),
    }
    write_json(output_path, result)

    try:
        torch.cuda.reset_peak_memory_stats(device_index)
        started = time.monotonic()
        model = ARC_Qwen2_5_VL_VideoForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            device_map=args.device,
            local_files_only=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        # ARC stores the speech encoder in its own checkpoint.  This downloads
        # only the feature-extractor metadata when it is not already cached.
        wav_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")

        audios, _ = load_audio_from_video(str(video_path))
        sample_rate = 16_000
        segment_length = sample_rate * 30
        segment_count = math.ceil(len(audios) / segment_length)
        if segment_count > 1:
            raise RuntimeError("bounded canary unexpectedly produced multiple audio segments")
        spectrogram = wav_processor(audios, sampling_rate=sample_rate, return_tensors="pt")["input_features"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {
                        "type": "text",
                        "text": (
                            "Describe the video as a timestamped timeline. Include scene changes, "
                            "speaker identities, and spoken English verbatim."
                        ),
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(model.device)
        inputs["values_audios"] = spectrogram.squeeze(0).to(model.device, dtype=torch.float16)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        result.update(
            {
                "status": "completed",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "peak_vram_mib": round(torch.cuda.max_memory_allocated(device_index) / 2**20, 1),
                "raw_narration": processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0],
            }
        )
    except Exception as error:  # Preserve a useful artifact for an expected P40 incompatibility.
        result.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        result["completed_at"] = iso_utc()
        write_json(output_path, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
