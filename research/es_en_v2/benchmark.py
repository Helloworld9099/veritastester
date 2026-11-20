#!/usr/bin/env python3
"""Measure ACTUAL (not inferred) CPU translation latency and peak RSS.

Only ordinary, synthetic, previously disclosed prompts are used. No
independent test case or private source text is opened. Text translation
ONLY; ASR and complete real-time audiovisual latency remain unverified.
"""
from __future__ import annotations

import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments/cpu_continuation_20261117"))
import train as core

ADAPTER_REL = "experiments/massive_stage3_20261117/saved/spanish/adapter_selected.safetensors"
ADAPTER_SHA256 = "14dddb5040afe49f8f05a1bfba3458d3321524b53b0fc05b359ad41a1ab75a4b"
TEST_PHRASES = [
    "Hola, ¿puedes ayudarme con la reserva?",
    "El concierto estuvo increíble ayer.",
    "No son quince dólares, son cincuenta.",
    "Llegaré tarde a la reunión de mañana.",
    "Quiero reservar una mesa para dos personas.",
    "No entiendo lo que quiso decir mi amigo.",
    "¿Podemos hablar más tarde, por favor?",
    "Mi tren sale a las ocho de la mañana.",
    "Voy a cambiar la contraseña esta noche.",
    "¿Puedes repetir la última parte?",
    "Estoy aprendiendo español con mis compañeros.",
    "No me gustó la película que vimos.",
    "Necesito pagar la factura antes del viernes.",
    "Mi hermano está trabajando ahora mismo.",
    "Este problema es difícil de explicar rápidamente.",
    "Gracias por ayudarme con la presentación.",
]


def quantile(data, percentile):
    ordered=sorted(data)
    return ordered[min(len(ordered)-1,round((len(ordered)-1)*percentile))]


def run(output: Path):
    if output.exists():
        raise FileExistsError("Benchmark report must not be overwritten")
    import torch
    from torch import nn
    from safetensors.torch import load_file
    from transformers import MarianMTModel, MarianTokenizer

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    spec=core.LANGUAGES["spanish"]
    adapter=ROOT/ADAPTER_REL
    if core.sha256(adapter)!=ADAPTER_SHA256:
        raise RuntimeError("Protected license-compatible Spanish adapter mismatch")
    root=ROOT/"research/es_en_v2/.temporary_pinned_model"
    if root.exists():
        raise FileExistsError("Do not reuse an unverified temporary base")
    began=time.perf_counter()
    core.download_base(root,ROOT/"research"/spec["manifest_rel"],spec)
    tokenizer=MarianTokenizer.from_pretrained(str(root),local_files_only=True)
    model=MarianMTModel.from_pretrained(str(root),local_files_only=True,
                                     use_safetensors=spec["use_safetensors"],weights_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    load_seconds=time.perf_counter()-began

    def predict():
        durations=[]
        lengths=[]
        with torch.no_grad():
            for text in TEST_PHRASES:
                start=time.perf_counter()
                inputs=tokenizer(text,return_tensors="pt",truncation=True,max_length=128)
                prediction=model.generate(**inputs,num_beams=4,max_new_tokens=96,use_cache=True)
                durations.append(time.perf_counter()-start)
                lengths.append(int(prediction.shape[-1]))
        return {"requests":len(durations),
                "p50_seconds":quantile(durations,.50),
                "p95_seconds":quantile(durations,.95),
                "mean_seconds":statistics.mean(durations),
                "max_seconds":max(durations),
                "generated_tokens_total":sum(lengths)}

    # Cache initialization before timed baseline, transparently record warm run.
    _=predict()
    base=predict()

    class LoRA(nn.Module):
        def __init__(self,base_layer):
            super().__init__()
            self.base=base_layer
            self.lora_A=nn.Parameter(torch.zeros(8,base_layer.in_features))
            self.lora_B=nn.Parameter(torch.zeros(base_layer.out_features,8))
        def forward(self,x):
            return self.base(x)+((x@self.lora_A.T)@self.lora_B.T)*2.0

    protocol=json.loads((ROOT/"research"/spec["protocol_rel"]).read_text())
    if len(protocol["target_modules"])!=36:
        raise RuntimeError("Unexpected adapter target count")
    for name in protocol["target_modules"]:
        parent,attribute=name.rsplit(".",1)
        module=model.get_submodule(name)
        if not isinstance(module,nn.Linear):
            raise RuntimeError("Unexpected LoRA target")
        setattr(model.get_submodule(parent),attribute,LoRA(module))
    actual=load_file(str(adapter))
    target_keys={k for k in model.state_dict() if ".lora_" in k}
    if set(actual)!=target_keys:
        raise RuntimeError("Adapter keys incompatible")
    state=model.state_dict()
    with torch.no_grad():
        for k,t in actual.items():
            if not torch.isfinite(t).all() or state[k].shape!=t.shape:
                raise RuntimeError("Invalid adapter tensor")
            state[k].copy_(t)
    adapted=predict()
    report={
        "status":"real_cpu_translation_only_benchmark",
        "base_model":spec["model"],"revision":spec["revision"],
        "license_compatible_adapter_sha256":ADAPTER_SHA256,
        "model_load_seconds":load_seconds,
        "base":base,"stage3_adapter":adapted,
        "prompts":len(TEST_PHRASES),
        "warm_cache":True,
        "num_beams":4,"max_new_tokens":96,
        "torch_version":torch.__version__,
        "cpu_threads":torch.get_num_threads(),
        "cpu_interop_threads":torch.get_num_interop_threads(),
        "peak_resident_set_size_kib_linux":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "adapter_trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
        "paid_compute_cost_usd":0,
        "complete_audio_to_captions_latency_measured":False,
        "meaning_preservation_accuracy_measured":False,
        "hardware_note":"Ephemeral GitHub Actions ubuntu-24.04 CPU runner; "
                        "not proof of identical local Windows/Mac performance.",
    }
    output.mkdir(parents=True,exist_ok=False)
    (output/"cpu_benchmark.json").write_text(json.dumps(report,indent=2))
    print("V2_REAL_CPU_TRANSLATION_BENCHMARK",json.dumps(report),flush=True)


if __name__=="__main__":
    run(ROOT/"research/es_en_v2/benchmarks")
