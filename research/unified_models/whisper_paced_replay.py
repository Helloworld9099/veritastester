"""Paced offline FLEURS DEV replay using one local Whisper-base checkpoint.

Each admitted prefix is re-encoded from scratch, with two decoders sharing
that encoder result. PCM arrives on a synthetic wall-clock schedule, not from
a device. This is not native incremental streaming or incumbent app QA.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
import platform
from pathlib import Path
from time import monotonic_ns
import wave

from .paced_audio import PacePolicy, run_paced_prefix
from .whisper_fleurs_dev import (
    DECODE, LANGUAGES, MAX_MANIFEST_BYTES, MAX_WAV_BYTES, PUBLISHER_REVISION,
    SHA, DiagnosticError, checked_wav, configure_cpu, edit_distance, model_pins,
    read_bounded, rss, unique_pairs, words,
)


_RATE = 16_000
_NS = 1_000_000_000


_PILOT_INPUTS = Path(__file__).with_name("fleurs_dev_pilot_inputs.json")
_PILOT_INPUTS_SHA256 = "c23b2aab3adbdb470590af0d1096e92c1a47cbd3a9c59963eb0de023f172255a"


def _verified_inputs(path, audio_root, expected_sha256, cap):
    # One immutable read: refuse a wrong pin before following *any* audio
    # locator. Reference strings are only operator-pinned by this replay.
    raw = read_bounded(path,MAX_MANIFEST_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise DiagnosticError("fixture_manifest_sha256_mismatch")
    pilot = read_bounded(_PILOT_INPUTS,MAX_MANIFEST_BYTES)
    if hashlib.sha256(pilot).hexdigest() != _PILOT_INPUTS_SHA256:
        raise DiagnosticError("published_pilot_input_pin_mismatch")
    official = json.loads(pilot,object_pairs_hook=unique_pairs)
    if official["split"] != "dev" or len(official["clips"]) != 8:
        raise DiagnosticError("published_pilot_scope_mismatch")
    rows = json.loads(raw,object_pairs_hook=unique_pairs)
    fields = {"id","language","wav","source","reference","sha256","duration"}
    if type(rows) is not list or len(rows) != len(official["clips"]):
        raise DiagnosticError("operator_manifest_does_not_match_pilot_count")
    indexed = {}
    for row in rows:
        if (type(row) is not dict or not fields <= row.keys() <= fields | {"original_sha256"}
                or type(row["id"]) is not str or type(row["language"]) is not str):
            raise DiagnosticError("unexpected_operator_manifest_shape")
        key = row["language"],row["id"]
        if key in indexed:
            raise DiagnosticError("duplicate_dev_pilot_id")
        indexed[key] = row
    # Check all declarations against the public pilot before opening selected WAVs.
    chosen = defaultdict(list)
    for entry in official["clips"]:
        key = entry["source_language"],entry["id"]
        if key not in indexed:
            raise DiagnosticError("dev_pilot_clip_identity_mismatch")
        row = indexed[key]
        if (row["sha256"] != entry["normalized_pcm16_wav_sha256"]
                or ("original_sha256" in row
                    and row["original_sha256"] != entry["publisher_wav_member_sha256"])
                or type(row["duration"]) not in (int,float) or row["duration"] != entry["seconds"]):
            raise DiagnosticError("dev_pilot_clip_pin_mismatch")
        for field in ("source","reference"):
            value = row[field]
            if type(value) is not str or not 0 < len(value) <= 8192 or not value.strip() or "\0" in value:
                raise DiagnosticError("invalid_operator_reference_or_transcript")
        if len(chosen[key[0]]) < cap:
            chosen[key[0]].append(row)
    selected = defaultdict(list)
    for language, records in chosen.items():
        for row in records:
            raw_wav = read_bounded(checked_wav(audio_root,row["wav"]),MAX_WAV_BYTES)
            if hashlib.sha256(raw_wav).hexdigest() != row["sha256"]:
                raise DiagnosticError("prepared_dev_wav_pin_mismatch")
            with wave.open(io.BytesIO(raw_wav),"rb") as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(), wav.getcomptype()) != (
                        1,2,_RATE,"NONE"):
                    raise DiagnosticError("expected_pcm16_mono_16000")
                count = wav.getnframes()
                pcm = wav.readframes(count)
            seconds = count / _RATE
            if not 0<count<=30*_RATE or len(pcm)!=count*2 or abs(seconds-row["duration"]) > .051:
                raise DiagnosticError("prepared_dev_wav_duration_mismatch")
            selected[language].append((row["source"],row["reference"],pcm,seconds))
    return selected


def _ms(ns):
    return round(ns / 1_000_000, 3) if ns is not None else None


def _percentile(values, n):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[(len(ordered)*n+99)//100-1]


def _pair_present(out):
    return out is not None and all(text.strip() for text in out)


def _duration_ns(samples):
    return (samples * _NS + _RATE - 1) // _RATE


def _capture_lead_ns(now, *, epoch, capture_end, prefix_samples):
    """Already arrived audio past this prefix, not the number of queued jobs."""
    return max(0, min(capture_end, now) - epoch - _duration_ns(prefix_samples))


def _revised(a,b):
    """A draft revises an existing normalized word prefix, not simple extension."""
    return any(words(new)[:len(words(old))] != words(old) for old,new in zip(a,b))


def _summarize_clip(run):
    draft = previous = final = final_pair = None
    revisions = delivered = 0
    start_leads, done_leads, waits, steps = [], [], [], []
    for event in run.events:
        wait = max(0,event.start_ns-event.scheduled_ns)
        start_lead = _capture_lead_ns(event.start_ns, epoch=run.epoch_ns,
                     capture_end=run.capture_end_ns, prefix_samples=event.prefix_samples)
        done_lead = _capture_lead_ns(event.finish_ns, epoch=run.epoch_ns,
                     capture_end=run.capture_end_ns, prefix_samples=event.prefix_samples)
        waits.append(wait)
        start_leads.append(start_lead)
        done_leads.append(done_lead)
        steps.append({
            "prefix_audio_ms":_ms(_duration_ns(event.prefix_samples)),
            "request_wait_ms":_ms(wait),
            "dual_compute_ms":_ms(event.finish_ns-event.start_ns),
            "capture_lead_at_dispatch_ms":_ms(start_lead),
            "capture_lead_at_finish_ms":_ms(done_lead),
            "final":event.final,"published":event.emitted,"disposition":event.disposition,
        })
        if not event.emitted:
            continue
        if event.final:
            final = event.finish_ns - run.capture_end_ns
            final_pair = event.out
        else:
            if draft is None:
                draft = event.finish_ns-run.epoch_ns
            delivered += 1
        if previous is not None and _revised(previous,event.out):
            revisions += 1
        previous = event.out
    return {
        "outcome":run.outcome,
        "audio_duration_ms":_ms(run.capture_end_ns-run.epoch_ns),
        "capture_start_to_first_paired_provisional_ms":_ms(draft),
        "capture_end_to_paired_stable_ms":_ms(final),
        "scheduled_provisionals":run.scheduled_provisionals,
        "skipped_provisional_releases":run.skipped_provisionals,
        "stale_computed_provisionals":sum(e.disposition=="stale" for e in run.events),
        "final_dispatch_wait_ms":run.timing_summary()["final_start_lag_ns"] / 1_000_000
                                 if any(e.final for e in run.events) else None,
        "published_provisionals":delivered,
        "publish_transitions_revising_prior_words":revisions,
        "observed_peak_request_wait_ms":_ms(max(waits,default=0)),
        "observed_peak_capture_lead_at_dispatch_ms":_ms(max(start_leads,default=0)),
        "observed_peak_capture_lead_at_finish_ms":_ms(max(done_leads,default=0)),
        "prefix_decodes":steps,
    }, final_pair


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
    torch.use_deterministic_algorithms(True)
    root = args.audio_root or args.manifest.parent
    if root.is_symlink() or not root.is_dir():
        raise DiagnosticError("invalid_audio_root")
    fixture_sha = args.expected_manifest_sha256
    rows = _verified_inputs(args.manifest,root.resolve(strict=True),fixture_sha,args.max_clips_per_direction)
    if sum(row[3] for clips in rows.values() for row in clips) > 80:
        raise DiagnosticError("total_audio_budget_exceeded")
    pins = model_pins(args.model_dir)
    started = monotonic_ns()
    processor = WhisperProcessor.from_pretrained(str(args.model_dir),local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model_dir),local_files_only=True,use_safetensors=True,weights_only=True,
        torch_dtype=torch.float32).to("cpu").eval()
    load_ms = _ms(monotonic_ns()-started)
    policy = PacePolicy(cadence_samples=args.cadence_ms*16,
                        deadline_ns=args.clip_deadline_seconds*_NS)
    encoder_calls = decoder_calls = 0
    def count_encoder(*_):
        nonlocal encoder_calls
        encoder_calls += 1
    hook = model.model.encoder.register_forward_hook(count_encoder)
    metric = sacrebleu.metrics.CHRF(char_order=6,word_order=0,beta=2)
    report = {
        "schema_version":1,"kind":"fleurs_dev_whisper_paced_prefix_replay",
        "scope":"publisher_development_only","manifest_sha256":fixture_sha,
        "pinned_pilot_audio_metadata_sha256":_PILOT_INPUTS_SHA256,
        "reference_identity":"operator_manifest_sha256_only_not_a_publisher_text_reattestation",
        "checkpoint_repository":"openai/whisper-base","checkpoint_publisher_revision":PUBLISHER_REVISION,
        "checkpoint_files":pins,"device":"cpu","torch_threads":args.threads,
        "torch_interop_threads":1,
        "libraries":{"torch":str(torch.__version__),"transformers":transformers.__version__,
                     "sacrebleu":sacrebleu.__version__},
        "host_platform":platform.system(),"host_machine":platform.machine(),
        "model_load_ms":load_ms,"cadence_ms":args.cadence_ms,
        "clip_deadline_seconds":args.clip_deadline_seconds,"decode":DECODE,
        "prefix_queue_policy":"single_worker_coalesce_ready_provisionals",
        "cold_warmup":"none","final_eof_policy":"always_request_full_clip_before_deadline",
        "paired_caption_rule":"both_source_and_translation_nonempty",
        "same_checkpoint_and_one_shared_encoder_per_completed_prefix":True,
        "reencodes_each_prefix":True,"native_incremental_streaming":False,
        "live_capture":False,"historical_or_live_incumbent_comparison":False,
        "cloud_or_gemini_used":False,"final_holdout_accessed":False,
        "independent_human_adequacy_measured":False,"promotion_eligible":False,
        "directions":{},"clips":[],
    }

    try:
        with torch.inference_mode():
            for language,clips in sorted(rows.items()):
                edits = ref_words = 0
                final_hyp,trans_ref, summaries = [],[],[]
                for index,(source,reference,pcm,_) in enumerate(clips,1):
                    samples = np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768.0
                    def decode(until, _final):
                        nonlocal decoder_calls
                        before = encoder_calls
                        inputs = processor(samples[:until], sampling_rate=_RATE,
                                           return_tensors="pt",return_attention_mask=True)
                        hidden = model.model.encoder(inputs.input_features).last_hidden_state
                        outputs = []
                        for task in ("transcribe","translate"):
                            generated = model.generate(
                                encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                                attention_mask=inputs.attention_mask,
                                language=LANGUAGES[language],task=task,**DECODE)
                            decoder_calls += 1
                            outputs.append(processor.batch_decode(
                                generated,skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)[0].strip())
                        if encoder_calls-before != 1:
                            raise DiagnosticError("unexpected_encoder_recompute_within_prefix")
                        return tuple(outputs)

                    pace = run_paced_prefix(len(samples), policy, decode, has_caption=_pair_present)
                    entry,final_pair = _summarize_clip(pace)
                    d = f"{language}->en"
                    report["clips"].append({"direction":d,"direction_clip_index":index,
                                            **entry,**rss()})
                    summaries.append(entry)
                    # An incomplete publication is not a zero-latency success.
                    # Score its missing full-clip outputs as empty explicitly.
                    a,b = final_pair or ("","")
                    expected = words(source)
                    edits += edit_distance(expected,words(a))
                    ref_words += len(expected)
                    final_hyp.append(b)
                    trans_ref.append(reference)

                provisional = [x["capture_start_to_first_paired_provisional_ms"] for x in summaries
                               if x["capture_start_to_first_paired_provisional_ms"] is not None]
                stable = [x["capture_end_to_paired_stable_ms"] for x in summaries
                          if x["capture_end_to_paired_stable_ms"] is not None]
                report["directions"][f"{language}->en"] = {
                    "clip_count":len(clips),"clips_with_paired_provisional":len(provisional),
                    "clips_with_paired_stable":len(stable),
                    "first_provisional_ms_p50":_percentile(provisional,50),
                    "first_provisional_ms_p95":_percentile(provisional,95),
                    "end_to_stable_ms_p50":_percentile(stable,50),
                    "end_to_stable_ms_p95":_percentile(stable,95),
                    "skipped_provisional_releases":sum(x["skipped_provisional_releases"] for x in summaries),
                    "stale_computed_provisionals":sum(x["stale_computed_provisionals"] for x in summaries),
                    "max_observed_capture_lead_at_dispatch_ms":max(
                        x["observed_peak_capture_lead_at_dispatch_ms"] for x in summaries),
                    "max_observed_capture_lead_at_finish_ms":max(
                        x["observed_peak_capture_lead_at_finish_ms"] for x in summaries),
                    "final_asr_wer_pct":round(edits*100/ref_words,6) if ref_words else None,
                    "final_translation_chrf":round(metric.corpus_score(final_hyp,[trans_ref]).score,6),
                    "missing_stable_outputs_scored_empty":len(clips)-len(stable),
                }
    finally:
        hook.remove()
    report["observed_encoder_forward_calls"] = encoder_calls
    report["observed_decoder_calls"] = decoder_calls
    report["translation_metric_signature"] = str(metric.get_signature())
    report["process_memory_after"] = rss()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest","model-dir","report"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--audio-root",type=Path)
    parser.add_argument("--expected-manifest-sha256",required=True)
    parser.add_argument("--threads",type=int,choices=(1,2,3,4),default=2)
    parser.add_argument("--cadence-ms",type=int,choices=range(1000,6001,500),default=3000)
    parser.add_argument("--max-clips-per-direction",type=int,choices=(1,2),default=1)
    parser.add_argument("--clip-deadline-seconds",type=int,choices=range(35,61,5),default=45)
    args = parser.parse_args()
    if not SHA.fullmatch(args.expected_manifest_sha256):
        parser.error("An independent lowercase SHA256 for the DEV manifest is required")
    if args.report.exists() or args.report.is_symlink():
        parser.error("Refusing to overwrite an existing report")
    try:
        output = run(args)
        fd = os.open(args.report,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            json.dump(output,stream,sort_keys=True,indent=2,allow_nan=False)
            stream.write("\n")
    except DiagnosticError as exc:
        parser.exit(2,f"Replay blocked: {exc}\n")
    except Exception as exc:
        parser.exit(2,f"Replay failed without exposing inputs: {type(exc).__name__}\n")
    print("Wrote aggregate paced offline DEV replay.")
    # A report may record negative evidence; no status promotes a model.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
