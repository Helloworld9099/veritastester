"""Bounded offline CPU diagnostic for one pinned Nemotron 3.5 ASR checkpoint.

Uses publisher development audio and non-overlapping mel chunks. Feature
extraction sees the whole clip; this does not measure causal microphone
latency, translation, or sustained streaming memory. No model or audio is
downloaded by this runner and no text, audio or token IDs are published.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
from time import perf_counter_ns
import wave

from .whisper_fleurs_dev import DiagnosticError, configure_cpu, edit_distance, rss, words


MODEL_REPO = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "ea30d66debe3740a08b573244286791d423d6b3e"
TRANSFORMERS_VERSION = "5.21.0"
LOOKAHEAD = 3
RATE = 16_000
MANIFEST_PIN = "bd95f7bed48191b41b3c9038efbeb7d690c862c1b538c542909db9db358650cf"
MODEL_PINS = {
    "config.json": "62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845",
    "generation_config.json": "993e5d4cb74a6fe9d6e7084a76b3313c1446740679be4676570c23b664fdc07e",
    "model.safetensors": "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174",
    "processor_config.json": "ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0",
    "tokenizer.json": "3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246",
    "tokenizer_config.json": "5c641c5b3f50702a60082690d27c1ce7fcb5a92c4a624793bcae0f21eda3d6e0",
}
CLIP_PINS = {
    "es-1615": "850c3a6378b51d7e3b39369c7e12df78114168a57747711ebad21001cff511fe",
    "es-1510": "c07f37ca057c50b8c48afe052cfefda86a8d2639c9352a2843ccca3a03c97c44",
    "es-1599": "a40e069c79affeac000af35be4b50bc7180209f4eac1129f4395f955e07574f9",
    "es-1589": "90936b29fe1e0b41d23be3f678fcbd0853cb59f7050709bb161684fb98536c29",
    "fr-1524": "da4ee9f5ddb3d8898d973cb6d70c6e7568c8825d2d177394c3762330902f3342",
    "fr-1570": "281c4d7e65d186abbd4659a0afe8f8e614900c9add588c6cbdd2447fdd8ee5ce",
    "fr-1533": "4e42e046b987eab5cf60073457dc9efac531e4a37ff26d995fd8900c50810166",
    "fr-1652": "d7c768432f38e8ad3c0a5cf59a57e2d291b6419e5ef8ba97ab832090273faf7b",
}
LOCALES = {"es": "es-US", "fr": "fr-FR"}


def _check(ok, code):
    if not ok:
        raise DiagnosticError(code)


def _real(path, *, is_dir=False):
    _check((path.is_dir() if is_dir else path.is_file())
           and not path.is_symlink() and path.resolve() == path.absolute(), "noncanonical_local_path")
    return path


def _read_small(path, cap):
    with _real(path).open("rb") as stream:
        data = stream.read(cap + 1)
    _check(0 < len(data) <= cap, "invalid_local_file_size")
    return data


def verify_audio(manifest_path, root):
    raw = _read_small(manifest_path, 100_000)
    _check(hashlib.sha256(raw).hexdigest() == MANIFEST_PIN, "dev_manifest_pin_mismatch")
    _real(root, is_dir=True)
    items = json.loads(raw)
    _check(type(items) is list and [item.get("id") for item in items] == list(CLIP_PINS),
           "unexpected_dev_clip_inventory")
    rows = defaultdict(list)
    for item in items:
        key = item["id"]
        filename = key + ".wav"
        lang = key.split("-")[0]
        _check(item["language"] == lang and item["wav"] == filename
               and item["sha256"] == CLIP_PINS[key], "unexpected_dev_audio_metadata")
        wav_bytes = _read_small(root / filename, 16 * RATE * 2 + 4096)
        _check(hashlib.sha256(wav_bytes).hexdigest() == CLIP_PINS[key], "dev_wav_pin_mismatch")
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            _check((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
                   == (1, 2, RATE, "NONE"), "invalid_dev_wav_format")
            samples = wav.getnframes()
            pcm = wav.readframes(samples)
        _check(0 < samples <= 16 * RATE and len(pcm) == 2 * samples, "invalid_dev_pcm_length")
        rows[lang].append((pcm, item["source"]))
    return rows


def verify_model(root):
    _real(root, is_dir=True)
    _check({entry.name for entry in root.iterdir()} == MODEL_PINS.keys(), "unexpected_model_file_inventory")
    report = {}
    for name, pin in MODEL_PINS.items():
        file = _real(root / name)
        size = file.stat().st_size
        cap = 3_000_000_000 if name.endswith(".safetensors") else 2_000_000
        _check(0 < size <= cap, "invalid_model_file_size")
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        _check(digest == pin, "model_file_pin_mismatch")
        report[name] = {"sha256": digest, "bytes": size}
    return report


def chunk_plan(filled, first, regular):
    """Half-open mel ranges and per-chunk padding (only the final range can pad)."""
    _check(all(type(n) is int and n > 0 for n in (filled, first, regular)),
           "invalid_mel_chunk_dimensions")
    cursor = 0
    size = first
    while cursor < filled:
        end = min(filled, cursor + size)
        yield (cursor, end, size - (end-cursor))
        cursor = end
        size = regular


class EncoderWitness:
    """Observed tensors at the actual encoder boundary, not generator intent."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0
        self.reuse = 0
        self.cache = None
        self.padding = None
        self.cached_steps = 0

    def __call__(self, _encoder, args, kwargs, result):
        _check(not args and self.calls < len(self.chunks), "unexpected_encoder_call")
        actual = kwargs.get("input_features")
        expected = self.chunks[self.calls]
        _check(actual is not None and actual.shape == expected.shape
               and actual.data_ptr() == expected.data_ptr(), "encoder_input_not_next_disjoint_mel_slice")
        _check(kwargs.get("use_cache") is True, "encoder_cache_not_enabled")
        if self.calls:
            _check(kwargs.get("past_key_values") is self.cache and kwargs.get("padding_cache") is self.padding,
                   "previous_acoustic_cache_not_passed")
            self.reuse += 1
        else:
            _check(kwargs.get("past_key_values") is None and kwargs.get("padding_cache") is None,
                   "unexpected_initial_cache")
        next_cache, next_padding = result.past_key_values, result.padding_cache
        _check(next_cache is not None and next_padding is not None, "acoustic_cache_missing")
        steps = int(next_cache.get_seq_length())
        _check(steps > self.cached_steps, "acoustic_cache_did_not_advance")
        self.cached_steps = steps
        self.cache, self.padding = next_cache, next_padding
        self.calls += 1

    def summary(self):
        _check(self.calls == len(self.chunks) and self.calls > 1, "incomplete_incremental_encoder_coverage")
        return {"encoder_forward_calls": self.calls, "acoustic_cache_reuses": self.reuse,
                "encoder_positions_processed": self.cached_steps}


def _ms(start):
    return round((perf_counter_ns()-start)/1e6, 3)


def _manual_greedy(model, chunks, prompt, torch, DecoderCache):
    """Decode from projected chunk features with the unchanged head and its cache."""
    blank = model.config.blank_token_id
    max_at_frame = model.config.max_symbols_per_step
    token = torch.tensor([[blank]], dtype=torch.long)
    prediction_cache = DecoderCache(model.config)
    kv = conv = None
    sequence = [blank]
    frame_advances = [0]
    for chunk in chunks:
        projected = model.get_audio_features(
            input_features=chunk, prompt_ids=prompt, use_cache=True,
            past_key_values=kv, padding_cache=conv,
            num_lookahead_tokens=LOOKAHEAD, output_attention_mask=False,
        )
        kv, conv = projected.past_key_values, projected.padding_cache
        # Don't accumulate projections across entire clip on this path.
        for frame in projected.pooler_output.split(1, dim=1):
            for j in range(max_at_frame):
                decoded = model(
                    encoder_outputs=type(projected)(pooler_output=frame),
                    decoder_input_ids=token, decoder_cache=prediction_cache,
                )
                token = decoded.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                predicted = int(token.item())
                sequence.append(predicted)
                advance = predicted == blank or j == max_at_frame - 1
                frame_advances.append(int(advance))
                if advance:
                    break
    return sequence, frame_advances


def _run_clip(model, processor, pcm, reference, locale, torch, np, DecoderCache):
    started = perf_counter_ns()
    wave_array = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    features = processor(wave_array, language=locale, sampling_rate=RATE, return_tensors="pt")
    tensor = features.input_features.to(device="cpu", dtype=torch.float32)
    nframes = int(features.attention_mask.sum())
    _check(tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[2] == 128
           and 0 < nframes <= tensor.shape[1], "unexpected_mel_shape")
    plan = list(chunk_plan(nframes, processor.num_mel_frames_first_audio_chunk,
                           processor.num_mel_frames_per_audio_chunk))
    slices = []
    for start, end, pad in plan:
        chunk = tensor[:, start:end, :]
        if pad:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, pad), value=math.log(2**-24))
        slices.append(chunk)
    feature_ms = _ms(started)

    def released():
        yield from slices

    witness = EncoderWitness(slices)
    handle = model.encoder.register_forward_hook(witness, with_kwargs=True)
    started = perf_counter_ns()
    try:
        native = model.generate(
            input_features=released(), prompt_ids=features.prompt_ids,
            num_lookahead_tokens=LOOKAHEAD, do_sample=False,
            max_new_tokens=sum(chunk.shape[1] for chunk in slices) * model.config.max_symbols_per_step,
        )
        native_ms = _ms(started)
    finally:
        handle.remove()
    native_evidence = witness.summary()
    original = native.sequences[0].tolist()
    original_advances = native.durations[0].tolist()
    _check(sum(original_advances) == native_evidence["encoder_positions_processed"],
           "vendor_decoder_did_not_consume_all_encoder_positions")
    # Release first run's saved acoustic cache before measuring the second run.
    del witness, native

    witness = EncoderWitness(slices)
    handle = model.encoder.register_forward_hook(witness, with_kwargs=True)
    started = perf_counter_ns()
    try:
        reconstructed, advances = _manual_greedy(model, slices, features.prompt_ids, torch, DecoderCache)
        manual_ms = _ms(started)
    finally:
        handle.remove()
    staged_evidence = witness.summary()
    _check(sum(advances) == staged_evidence["encoder_positions_processed"],
           "staged_decoder_did_not_consume_all_encoder_positions")
    hypothesis = processor.decode(original, skip_special_tokens=True)
    expected, observed = words(reference), words(hypothesis)
    _check(bool(expected) and bool(observed), "empty_reference_or_transcription")
    return {
        "samples": len(pcm)//2, "locale_prompt": locale, "mel_frames": nframes,
        "padded_terminal_mel_frames": plan[-1][2], "feature_extraction_ms": feature_ms,
        "vendor_streaming": {**native_evidence, "elapsed_ms": native_ms},
        "staged_shared_feature_path": {**staged_evidence, "elapsed_ms": manual_ms},
        "identical_blank_inclusive_token_ids": original == reconstructed,
        "identical_frame_advances": original_advances == advances,
        "asr_word_errors": edit_distance(expected, observed), "reference_words": len(expected),
    }


def run(args):
    configure_cpu(args.threads)
    rows = verify_audio(args.manifest, args.audio_root)
    pins = verify_model(args.model_dir)
    import numpy as np
    import torch
    import transformers
    from transformers import AutoProcessor, Nemotron3_5AsrForRNNT
    from transformers.models.nemotron3_5_asr.generation_nemotron3_5_asr import Nemotron3_5AsrRNNTDecoderCache

    _check(transformers.__version__ == TRANSFORMERS_VERSION, "untested_transformers_version")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    started = perf_counter_ns()
    processor = AutoProcessor.from_pretrained(str(args.model_dir), trust_remote_code=False, local_files_only=True)
    processor.set_num_lookahead_tokens(LOOKAHEAD)
    model = Nemotron3_5AsrForRNNT.from_pretrained(
        str(args.model_dir), local_files_only=True, use_safetensors=True, weights_only=True,
        dtype=torch.float32,
    ).to("cpu").eval()
    load_ms = _ms(started)
    _check(model.device.type == "cpu" and model.dtype == torch.float32, "unexpected_model_placement")

    results = {}
    with torch.inference_mode():
        for lang, items in sorted(rows.items()):
            clips = [
                _run_clip(model, processor, pcm, source, LOCALES[lang], torch, np,
                          Nemotron3_5AsrRNNTDecoderCache)
                for pcm, source in items[:args.clips_per_language]
            ]
            err = sum(c["asr_word_errors"] for c in clips)
            n = sum(c["reference_words"] for c in clips)
            results[lang] = {"clips": clips, "wer_pct": round(100*err/n, 4)}
    parity = all(c["identical_blank_inclusive_token_ids"] and c["identical_frame_advances"]
                 for result in results.values() for c in result["clips"])
    return {
        "schema_version": 1, "kind": "nemotron_cpu_asr_cached_mel_dev_smoke",
        "status": "observed_asr_cache_and_token_parity" if parity else "asr_parity_failed",
        "scope": "publisher_development_only", "manifest_sha256": MANIFEST_PIN, "verified_wavs": len(CLIP_PINS),
        "checkpoint_repository": MODEL_REPO, "checkpoint_revision": MODEL_REVISION, "checkpoint_files": pins,
        "cpu_threads": args.threads, "torch": str(torch.__version__), "transformers": transformers.__version__,
        "host_os": platform.system(), "host_machine": platform.machine(),
        "model_load_ms": load_ms, "final_process_memory": rss(), "lookahead_setting": LOOKAHEAD,
        "nominal_model_chunk_ms": processor.streaming_latency_ms,
        "front_end": "whole_clip_mel_then_disjoint_encoder_slices_with_final_mel_padding",
        "audio_paced": False, "live_latency_measured": False, "sustained_memory_measured": False,
        "translation_head_present": False, "model_trained": False, "promotion_eligible": False,
        "asr": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("model-dir", "manifest", "audio-root", "report"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--clips-per-language", type=int, choices=(1, 4), default=4)
    args = parser.parse_args(argv)
    _check(not args.report.exists() and not args.report.is_symlink(), "existing_report_refused")
    try:
        report = run(args)
        # No overwrite, even if a different process creates the path mid-run.
        fd = os.open(args.report, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except DiagnosticError as exc:
        parser.exit(2, f"Smoke blocked: {exc}\n")
    except Exception as exc:
        parser.exit(2, f"Smoke failed without exposing inputs: {type(exc).__name__}\n")
    return int(report["status"] == "asr_parity_failed")


if __name__ == "__main__":
    raise SystemExit(main())
