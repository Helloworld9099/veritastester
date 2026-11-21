#!/usr/bin/env python3
"""Local-only bounded FLEURS DEV diagnostic; never emits audio or caption text."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import io
import json
import math
import os
import platform
from pathlib import Path
import re
from time import perf_counter_ns
from types import SimpleNamespace
import unicodedata
import wave

LANGUAGES = {"es": "spanish", "fr": "french"}
LANGUAGE_ALIASES = {"es": "es", "es_es": "es", "es-es": "es", "spanish": "es",
                    "fr": "fr", "fr_fr": "fr", "fr-fr": "fr", "french": "fr"}
SHA = re.compile(r"[a-f0-9]{64}\Z")
MODEL_FILES = frozenset({"config.json", "generation_config.json", "model.safetensors",
    "preprocessor_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json",
    "merges.txt", "normalizer.json", "added_tokens.json", "special_tokens_map.json"})
MAX_WAV_BYTES = 16_000 * 2 * 31 + 4096
MAX_MANIFEST_BYTES = 1 << 20
PUBLISHER_REVISION = "e37978b90ca9030d5170a5c07aadb050351a65bb"
PUBLISHER_SAFETENSORS_SHA = "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b"
PUBLISHER_FILE_PINS = {
    'added_tokens.json': '9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1',
    'config.json': 'a153c53883a6799b6f056b4a8d1a515c9926d03994682ba88a7616618d7da0c1',
    'generation_config.json': '444b3f636d2fff89dd9ecf549e2a085b61f7ff0fa0246d4628bac6a3b8cc9ba4',
    'merges.txt': '2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6',
    'model.safetensors': '07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b',
    'normalizer.json': 'bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd',
    'preprocessor_config.json': '9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666',
    'special_tokens_map.json': 'e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691',
    'tokenizer.json': '27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566',
    'tokenizer_config.json': '2a4c4281cf9f51ac6ccc406fdc711a087afe6530f671fa7b80953edc498275ce',
    'vocab.json': '8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9',
}

DECODE = {"num_beams": 1, "max_new_tokens": 128, "return_timestamps": False,
          "do_sample": False}


class DiagnosticError(ValueError):
    pass


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticError("duplicate_json_key")
        result[key] = value
    return result


def read_bounded(path, limit):
    if path.is_symlink() or not path.is_file():
        raise DiagnosticError("expected_local_regular_file")
    with path.open("rb") as file:
        data = file.read(limit + 1)
    if not data or len(data) > limit:
        raise DiagnosticError("local_file_size_outside_limit")
    return data


def checked_wav(root, locator):
    if type(locator) is not str or not locator or not locator.lower().endswith(".wav"):
        raise DiagnosticError("invalid_wav_locator")
    raw_path = Path(locator)
    if ".." in raw_path.parts:
        raise DiagnosticError("invalid_wav_locator")
    path = raw_path if raw_path.is_absolute() else root / raw_path
    candidate = path.resolve(strict=True)
    if not candidate.is_relative_to(root) or candidate != path.absolute() or path.is_symlink():
        raise DiagnosticError("wav_escapes_local_root")
    return path


def manifest(path, audio_root, cap):
    raw = read_bounded(path, MAX_MANIFEST_BYTES)
    data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs)
    if type(data) is not list or not data or len(data) > 48:
        raise DiagnosticError("invalid_manifest_size")
    used = set()
    rows = defaultdict(list)
    for item in data:
        required = {"id", "language", "wav", "source", "reference", "sha256", "duration"}
        if type(item) is not dict or not required <= item.keys() <= required | {"original_sha256"}:
            raise DiagnosticError("unexpected_dev_manifest_shape")
        if "original_sha256" in item and (type(item["original_sha256"]) is not str
                or not SHA.fullmatch(item["original_sha256"])):
            raise DiagnosticError("invalid_original_clip_checksum_metadata")
        key, lang = item["id"], item["language"]
        if type(key) not in (int, str) or not str(key) or type(lang) is not str:
            raise DiagnosticError("invalid_clip_identity_or_language")
        language = LANGUAGE_ALIASES.get(lang.lower())
        identity = (language, str(key))
        if language is None or identity in used:
            raise DiagnosticError("unsupported_or_duplicate_clip_language")
        used.add(identity)
        for field in ("source", "reference"):
            value = item[field]
            if type(value) is not str or not 0 < len(value) <= 8192 or not value.strip() or "\0" in value:
                raise DiagnosticError("invalid_reference_or_transcript")
        pin = item["sha256"]
        duration = item["duration"]
        if type(pin) is not str or not SHA.fullmatch(pin):
            raise DiagnosticError("missing_clip_checksum")
        if type(duration) not in (int, float) or not 0.01 <= duration <= 30 or not math.isfinite(duration):
            raise DiagnosticError("invalid_declared_clip_duration")
        if len(rows[language]) >= cap:
            continue
        audio = read_bounded(checked_wav(audio_root,item["wav"]), MAX_WAV_BYTES)
        if hashlib.sha256(audio).hexdigest() != pin:
            raise DiagnosticError("clip_checksum_mismatch")
        with wave.open(io.BytesIO(audio), "rb") as wav:
            if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getcomptype()) != (
                    1, 2, 16_000, "NONE"):
                raise DiagnosticError("expected_pcm16_mono_16000")
            samples = wav.getnframes()
            seconds = samples / 16_000
            pcm = wav.readframes(samples)
        if not 0 < samples <= 480_000 or len(pcm) != samples*2 or abs(duration-seconds) > .051:
            raise DiagnosticError("clip_length_mismatch")
        rows[language].append((item["source"], item["reference"], pcm, seconds))
    if set(rows) != LANGUAGES.keys():
        raise DiagnosticError("both_es_and_fr_dev_directions_required")
    return rows, hashlib.sha256(raw).hexdigest()


def model_pins(folder):
    if folder.is_symlink() or not folder.is_dir():
        raise DiagnosticError("expected_local_model_directory")
    entries = {path.name: path for path in folder.iterdir()}
    if not {"config.json","preprocessor_config.json","model.safetensors"} <= entries.keys():
        raise DiagnosticError("local_whisper_safetensors_missing")
    if any(p.is_symlink() or not p.is_file() for n,p in entries.items() if n in MODEL_FILES):
        raise DiagnosticError("unsafe_local_model_file")
    if any(n.endswith((".py", ".bin", ".pt", ".pth", ".ckpt")) for n in entries):
        raise DiagnosticError("unexpected_code_or_alternate_model_weights")
    try:
        config = json.loads(read_bounded(entries["config.json"], 65536), object_pairs_hook=unique_pairs)
        if (config.get("model_type") != "whisper" or config.get("d_model") != 512
                or config.get("encoder_layers") != 6 or config.get("decoder_layers") != 6
                or config.get("is_encoder_decoder") is not True or "auto_map" in config):
            raise DiagnosticError("local_model_not_whisper_base_architecture")
    except AttributeError:
        raise DiagnosticError("invalid_whisper_config") from None
    pins = {}
    for name,path in sorted(entries.items()):
        if name not in MODEL_FILES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(1<<20),b""):
                digest.update(block)
        pins[name] = {"sha256":digest.hexdigest(),"bytes":path.stat().st_size}
    if set(entries) != set(PUBLISHER_FILE_PINS) or {
        name: row["sha256"] for name, row in pins.items()
    } != PUBLISHER_FILE_PINS:
        raise DiagnosticError("local_whisper_files_do_not_match_publisher_pins")
    return pins


def words(text):
    """Casefold/NFC; punctuation and whitespace are word boundaries, retain accents."""
    clean = "".join(char if unicodedata.category(char)[0] not in "PZ" else " "
                    for char in unicodedata.normalize("NFC",text).casefold())
    return clean.split()


def edit_distance(reference, observed):
    prev = list(range(len(observed)+1))
    for i,w in enumerate(reference,1):
        row = [i]
        for j,x in enumerate(observed,1):
            row.append(min(row[-1]+1,prev[j]+1,prev[j-1]+(w != x)))
        prev = row
    return prev[-1]


def ms(start):
    return round((perf_counter_ns()-start)/1e6,3)


def rss():
    try:
        with Path("/proc/self/status").open(encoding="ascii") as file:
            values = {row.split(":")[0]:int(row.split()[1]) for row in file
                      if row.startswith(("VmRSS:", "VmHWM:"))}
        return {"process_rss_kib":values.get("VmRSS"),
                "process_peak_rss_kib_so_far": values.get("VmHWM")}
    except (OSError, ValueError, IndexError):
        return {"process_rss_kib":None,"process_peak_rss_kib_so_far":None}


def configure_cpu(threads):
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    for name in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def run(args):
    configure_cpu(args.threads)
    import numpy as np
    import torch
    import transformers
    import sacrebleu
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers.modeling_outputs import BaseModelOutput
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    root = args.audio_root or args.manifest.parent
    if root.is_symlink() or not root.is_dir():
        raise DiagnosticError("invalid_audio_root")
    rows, manifest_pin = manifest(args.manifest,root.resolve(strict=True),args.max_clips_per_direction)
    pins = model_pins(args.model_dir)
    start = perf_counter_ns()
    processor = WhisperProcessor.from_pretrained(str(args.model_dir),local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model_dir), local_files_only=True, use_safetensors=True, weights_only=True,
        torch_dtype=torch.float32).to("cpu").eval()
    load_ms = ms(start)
    baseline_queue = defaultdict(list)
    report = {
        "schema_version":1, "kind":"fleurs_local_whisper_base_dev_diagnostic",
        "scope":"publisher_development_only",
        "manifest_sha256":manifest_pin,
        "checkpoint_architecture":"whisper_base_multilingual",
        "checkpoint_repository":"openai/whisper-base",
        "checkpoint_publisher_revision":PUBLISHER_REVISION,
        "checkpoint_files":pins,
        "host_platform":platform.system(),
        "host_machine":platform.machine(),
        "device":"cpu","torch_threads":args.threads,"torch_interop_threads":1,
        "libraries":{"torch":str(torch.__version__),"transformers":transformers.__version__,
                     "sacrebleu":sacrebleu.__version__},
        "model_load_ms":load_ms,"same_whisper_instance_two_decode_tasks":True,
        "decode":DECODE,"one_audio_processing_pass_per_clip":True,
        "encoder_computations_per_clip":1,"decoder_generation_passes_per_clip":2,
        "audio_to_partial_caption":"not_measured","true_streaming":False,
        "independent_human_meaning_accuracy":"not_measured","promotion_eligible":False,
        "app_incumbent_full_pipeline_measured":False,"clip_statistics":[],
        "directions":{},
    }
    metric = sacrebleu.metrics.CHRF(char_order=6,word_order=0,beta=2)
    encoder_calls = 0
    def note_encoder(*_):
        nonlocal encoder_calls
        encoder_calls += 1
    hook = model.model.encoder.register_forward_hook(note_encoder)
    with torch.inference_mode():
        for language in sorted(rows):
            ref_words = asr_edits = 0
            translations,references = [],[]
            for idx,(source,ref,pcm,seconds) in enumerate(rows[language],1):
                calls_before = encoder_calls
                t_all = t = perf_counter_ns()
                samples = np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768.0
                inputs = processor(samples,sampling_rate=16_000,return_tensors="pt",
                                   return_attention_mask=True)
                preprocess_ms = ms(t)
                t = perf_counter_ns()
                states = model.model.encoder(inputs.input_features)
                shared_encoder_ms = ms(t)
                decoded, elapsed = {},{}
                for task in ("transcribe","translate"):
                    t = perf_counter_ns()
                    out = model.generate(
                        encoder_outputs=BaseModelOutput(last_hidden_state=states.last_hidden_state),
                        attention_mask=inputs.attention_mask,
                        language=LANGUAGES[language], task=task, **DECODE)
                    decoded[task] = processor.batch_decode(out, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)[0].strip()
                    elapsed[task] = ms(t)
                full_ms = ms(t_all)
                if encoder_calls - calls_before != 1:
                    raise DiagnosticError("encoder_forward_count_did_not_match_contract")
                ref_tokens = words(source)
                if not ref_tokens:
                    raise DiagnosticError("empty_normalized_asr_reference")
                edits = edit_distance(ref_tokens,words(decoded["transcribe"]))
                asr_edits += edits; ref_words += len(ref_tokens)
                translations.append(decoded["translate"]); references.append(ref)
                if args.historical_repo is not None:
                    baseline_queue[language].append((decoded["transcribe"],ref))
                report["clip_statistics"].append({
                    "direction":f"{language}->en","direction_clip_index":idx,
                    "audio_seconds":round(seconds,6),
                    "preprocess_ms":preprocess_ms,"shared_encoder_ms":shared_encoder_ms,
                    "transcribe_decode_ms":elapsed["transcribe"],"translate_decode_ms":elapsed["translate"],
                    "dual_result_ms":full_ms,"asr_edit_count":edits,"asr_reference_words":len(ref_tokens),
                    "asr_empty":not bool(decoded["transcribe"]),
                    "translation_empty":not bool(decoded["translate"]),
                    **rss(),
                })
            report["directions"][f"{language}->en"] = {
                "clips":len(references),"asr_edit_count":asr_edits,"asr_reference_words":ref_words,
                "asr_wer_pct":round(asr_edits*100/ref_words,6),
                "translation_chrf":round(metric.corpus_score(translations,[references]).score,6),
            }
    hook.remove()
    report["observed_encoder_forward_calls"] = encoder_calls
    report["translation_metric_signature"] = str(metric.get_signature())
    report["asr_wer_normalization"] = "NFC casefold punctuation-to-space accents-retained corpus WER"
    report["whisper_process_memory_after"] = rss()
    if args.historical_repo is not None:
        del model, processor, states, inputs, out
        gc.collect()
        report["historical_marian_asr_text_proxy"] = proxy(args, baseline_queue, metric)
    return report


def proxy(args, inputs, metric):
    """Opt-in proxy, not the live incumbent and no cloud provider call."""
    import sys
    repo = args.historical_repo.resolve(strict=True)
    if repo != Path(__file__).resolve().parents[2]:
        raise DiagnosticError("historical_proxy_must_use_this_verified_checkout")
    sys.path.insert(0, str(repo))
    from research.unified_models.historical.runtime import HistoricalMarianBaseline
    engine = HistoricalMarianBaseline(cache_dir=args.historical_base_cache.resolve(strict=True),
                artifact_root=args.historical_artifact_root.resolve(strict=True),
                historical_replay=True)
    results = {}
    for language,rows in sorted(inputs.items()):
        hyp,ref,timings = [],[],[]
        failed = 0
        for transcribed,gold in rows:
            t = perf_counter_ns()
            try:
                translation = engine.translate(SimpleNamespace(text=transcribed,source=language,
                            target="en",context_enabled=False)) if transcribed else ""
            except (ValueError, RuntimeError, OSError):
                translation = ""
                failed += 1
            timings.append(ms(t))
            hyp.append(translation); ref.append(gold)
        results[f"{language}->en"] = {"clips":len(ref),
            "failed_translation_calls_scored_empty":failed,
            "translation_chrf":round(metric.corpus_score(hyp,[ref]).score,6),
            "translation_call_ms_includes_first_load":timings}
        engine.unload()
    return {"status":"measured_offline_proxy_not_live_app","directions":results,
            "historical_adapter_pins":engine.routes()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--audio-root",type=Path)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--threads",type=int,choices=range(1,5),default=2)
    parser.add_argument("--max-clips-per-direction",type=int,choices=range(1,17),default=6)
    parser.add_argument("--historical-repo",type=Path)
    parser.add_argument("--historical-base-cache",type=Path)
    parser.add_argument("--historical-artifact-root",type=Path)
    args = parser.parse_args()
    selected = (args.historical_repo,args.historical_base_cache,args.historical_artifact_root)
    if any(selected) and not all(selected):
        parser.error("All three historical replay roots must be provided together")
    if args.report.exists() or args.report.is_symlink():
        parser.error("Refusing to overwrite an existing report")
    try:
        result = run(args)
        fd = os.open(args.report, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd,"w",encoding="utf-8") as file:
            json.dump(result,file,sort_keys=True,indent=2,allow_nan=False)
            file.write("\n")
    except DiagnosticError as exc:
        parser.exit(2,f"Diagnostic blocked: {exc}\n")
    except Exception as exc:
        parser.exit(2,f"Diagnostic failed without exposing inputs: {type(exc).__name__}\n")
    print("Wrote aggregate local DEV diagnostic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
