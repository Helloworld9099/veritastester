"""Explicit historical specialist replay against the frozen, byte-pinned M0 DEV dataset.

Run from the detached PR43 checkout (pre) or the unified checkout historical package (post). It never exercises the proposed one-model interface.
This script never modifies a checkout or model/fixture files, and reads checkpoint
weights only from explicit, local roots. The fixture does not stand in for QA on application audio or translation meaning.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata as md
import inspect
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

FIXED_SCOPE = ('en-es','es-en','fr-en')
FROZEN_FIXTURE_SHA256 = 'd5fce03b8c49d9dd071da0619661ea57106c6847b750101cb4d0693b3ae51a0c'
FROZEN_MANIFEST_SHA256 = '7dc7be2a0df65f8507a67f9cf1e61b558fefdecad3bb81d8552cd4b62b5d36f4'
SOURCE_M0_HEAD = 'ca755aec8f7111f03145e18ef9948cd6127aaec0'

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()

def text_sha(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def deny(*args, **kwargs):
    raise RuntimeError("No network downloads allowed during deterministic DEV replay")

def method_def(obj, name):
    return Path(inspect.getsourcefile(getattr(type(obj), name)) or inspect.getfile(type(obj))).resolve()

def bad_route_checks(engine):
    for source,target in [('zh','en'),('en','zh'),('cmn','en'),('en','cmn'),('en','fr'),('fr','es'),('auto','en')]:
        try: engine.registry.select(source,target)
        except (ValueError, KeyError, NotImplementedError): continue
        raise AssertionError('Unexpected active route: '+source+'-'+target)

def main():
    assert __debug__, 'Checks require unoptimized Python'
    a=argparse.ArgumentParser()
    a.add_argument('--phase',choices=('pre','post'), required=True)
    a.add_argument('--attempt',type=int,default=1)
    a.add_argument('--cache',type=Path,required=True)
    a.add_argument('--fixtures',type=Path,required=True)
    a.add_argument('--manifest',type=Path,required=True)
    a.add_argument('--receipt',type=Path,required=True)
    a.add_argument('--artifact-root',type=Path,help='Detached PR43 checkout for exact, read-only adapter assets (required in post)')
    a.add_argument('--expect-registry-path',help='Assert implementation file of select() relative to candidate repository')
    a.add_argument('--compare-pre-receipt',type=Path,help='Optional prior PRE result; gate rollback parity and record comparison timings')
    x=a.parse_args()
    repo=Path.cwd().resolve()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if x.phase=='pre':
        assert head==SOURCE_M0_HEAD and x.artifact_root is None,head
        artifact_root=repo
    else:
        assert x.artifact_root is not None, '--artifact-root required for post'
        artifact_root=x.artifact_root.resolve()
        assert artifact_root!=repo, 'Post replay source and artifact roots must be different'
        artifact_head=subprocess.check_output(['git','-C',str(artifact_root),'rev-parse','HEAD'],text=True).strip()
        assert artifact_head==SOURCE_M0_HEAD,artifact_head
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path[:0]=[str(repo),str(repo/'src')]
    fixture_hash=sha(x.fixtures)
    assert fixture_hash==FROZEN_FIXTURE_SHA256
    assert sha(x.manifest)==FROZEN_MANIFEST_SHA256
    manifest=json.loads(x.manifest.read_text())
    assert manifest['fixture_sha256']==fixture_hash and manifest['source_commit']==SOURCE_M0_HEAD
    rows=[json.loads(s) for s in x.fixtures.read_text().splitlines() if s]
    assert len(rows)==manifest['fixture_rows']==18 and len({r['id'] for r in rows})==18
    assert set(manifest['directions'])=={r['direction'] for r in rows}==set(FIXED_SCOPE)
    assert all(r['source_lang']+'-'+r['target_lang']==r['direction'] for r in rows)
    for file,h in manifest['source_pins'].items(): assert sha(artifact_root/file)==h,(file,h)
    import torch
    import huggingface_hub
    from bridgecaption.pipeline import Pipeline
    from bridgecaption.provider import TranslationRequest
    from bridgecaption.providers import PROVIDERS
    assert Path(inspect.getfile(Pipeline)).resolve().is_relative_to(repo)
    assert 'local_marian' not in PROVIDERS
    module_prefix = ('research/unified_models' if x.phase=='pre'
                     else 'research/unified_models/historical')
    registry_file = module_prefix + '/registry.json'
    sources=[module_prefix+'/registry.py',registry_file,
             module_prefix+'/runtime.py','src/bridgecaption/provider.py',
             'src/bridgecaption/providers.py','src/bridgecaption/pipeline.py']
    sources += [('research/es_en_v2' if x.phase=='pre' else 'research/unified_models')+'/inference_safety.py']
    observed_sources={f:sha(repo/f) for f in sources}
    versions={k:md.version(k) for k in manifest['method']['versions']}
    assert versions==manifest['method']['versions'],versions
    torch.set_num_threads(manifest['method']['torch_threads'])
    torch.set_num_interop_threads(manifest['method']['torch_interop_threads'])
    torch.manual_seed(manifest['method']['torch_seed'])
    torch.use_deterministic_algorithms(True)
    with ExitStack() as guard:
        guard.enter_context(patch.object(huggingface_hub,'hf_hub_download',deny))
        guard.enter_context(patch.object(huggingface_hub,'snapshot_download',deny))
        guard.enter_context(patch('socket.create_connection',deny))
        guard.enter_context(patch('socket.socket.connect',deny))
        if x.phase=='pre':
            from research.unified_models.runtime import SharedMarianResearchTranslator as Baseline
            from research.unified_models.registry import ModelRegistry
            options = {}
        else:
            from research.unified_models.historical.runtime import HistoricalMarianBaseline as Baseline
            from research.unified_models.historical.registry import ModelRegistry
            options = {'artifact_root': artifact_root, 'historical_replay': True}
        registry=ModelRegistry()
        assert registry.root==repo
        assert Path(inspect.getfile(Baseline)).resolve().is_relative_to(repo)
        engine=Baseline(registry=registry,cache_dir=x.cache.resolve(),
                                             retry_repetition=True,**options)
        registry_path=method_def(engine.registry,'select').relative_to(repo).as_posix()
        translate_path=method_def(engine,'translate').relative_to(repo).as_posix()
        if x.expect_registry_path:
            assert registry_path==x.expect_registry_path,(registry_path,x.expect_registry_path)
        if x.phase=='post':
            assert engine.artifact_root.resolve()==artifact_root
        attest={'legacy_artifact_root_revision':SOURCE_M0_HEAD,
                'registry_is_local_candidate_instance':engine.registry is registry}
        bad_route_checks(engine)
        pins={}
        if x.phase=='post':
            for d in FIXED_SCOPE:
                entry=registry.select(*d.split('-'))
                registry.verify_base(entry,x.cache.resolve())
                registry.verify_adapter(entry,artifact_root)
                if entry.rollback_adapter:
                    registry.verify_adapter(entry,artifact_root,use_previous=True)
                pins[d]={'model':entry.model,'revision':entry.revision,
                         'active_sha256':entry.active_adapter['sha256'] if entry.active_adapter else None,
                         'rollback_sha256':entry.rollback_adapter['sha256'] if entry.rollback_adapter else None,
                         'model_status':entry.rights_review_status,
                         'tokenizer_hashes':{t:entry.files[t]['sha256'] for t in ('source.spm','target.spm')}}
        else:
            pins=registry.verify_sources()
        assert set(pins)==set(FIXED_SCOPE)
        pipe=Pipeline(translator=engine)
        assert not engine.supports_context
        assert engine.loaded_model_count==0
        timings={}; results=[]; rollback={}
        for d in FIXED_SCOPE:
            frozen=manifest['directions'][d]
            src,tgt=d.split('-')
            spec=engine.registry.select(src,tgt)
            assert spec.model_id==d
            assert pins[d]['model']==frozen['base_model']==spec.model
            assert pins[d]['revision']==frozen['base_revision']==spec.revision
            assert pins[d]['active_sha256']==frozen['active_adapter_sha256']
            for t in ('source.spm','target.spm'):
                assert pins[d]['tokenizer_hashes'][t]==frozen['base_file_sha256'][t]
            old_rights=frozen['rights_status']
            new_rights=pins[d]['model_status']
            # M1 clarifies an unresolved restriction on this exact Spanish
            # ancestry; no research-only restriction can be relaxed.
            explicit_rights_clarification=(
                x.phase=='post' and d=='es-en'
                and old_rights=='research_only_pending_commercial_provenance_review'
                and new_rights=='research_only_pending_europarl_commercial_provenance_review'
            )
            assert new_rights==old_rights or explicit_rights_clarification, (d,old_rights,new_rights)
            for name,h in frozen['base_file_sha256'].items():
                assert '/' not in name and sha(x.cache/d/spec.revision/name)==h,(d,name)
            assert engine.loaded_model_count==0
            cases=[r for r in rows if r['direction']==d]
            assert len(cases)==6
            t0=time.perf_counter_ns()
            pipe.translate(cases[0]['source'],source=src,target=tgt)
            cold_load_ms=(time.perf_counter_ns()-t0)/1e6
            warm=[]
            for r in cases:
                t0=time.perf_counter_ns()
                cap=pipe.translate(r['source'],source=src,target=tgt)
                warm.append((time.perf_counter_ns()-t0)/1e6)
                assert cap.context_info.startswith('disabled: 0 prior utterance(s)')
                assert not pipe.history.snapshot() and engine.loaded_model_count==1
                actual=cap.translated
                expected=r['baseline_output']
                expected_id=frozen['active_adapter_sha256'] or spec.revision
                assert engine.last_run['direction']==d and engine.last_run['version_sha256']==expected_id
                assert not engine.last_run['rollback_selected']
                results.append({'id':r['id'],'direction':d,'output_sha256':text_sha(actual),
                                'expected_sha256':text_sha(expected),
                                'match':actual.encode('utf-8')==expected.encode('utf-8'),
                                'repetition_retry_matched':bool(r['generation_retry_for_loop'])==engine.last_run['generation_retry_for_loop']})
            timings[d]={'cold_load_plus_first_decoding_ms':cold_load_ms,'warm_all_ms':warm,
                         'warm_p50_ms':round(statistics.median(warm),3),'warm_p95_ms':round(sorted(warm)[math.ceil(len(warm)*.95)-1],3)}
            engine.unload()
        # Rollbacks are not evaluated against the active fixture. This only
        # checks distinct pinned historical route selection and actual loading.
        for d in ('es-en','fr-en'):
            src,tgt=d.split('-')
            first=next(r for r in rows if r['direction']==d)
            got=engine.translate(TranslationRequest(text=first['source'],source=src,target=tgt),use_previous=True)
            obs=engine.last_run
            assert got.strip() and obs['rollback_selected'] and obs['direction']==d
            assert obs['version_sha256']==pins[d]['rollback_sha256']!=pins[d]['active_sha256']
            assert engine.loaded_model_count==1
            rollback[d]={'output_sha256':text_sha(got),'adapter_sha256':obs['version_sha256']}
            engine.unload()
    ok=all(item['match'] and item['repetition_retry_matched'] for item in results)
    comparison=None
    if x.compare_pre_receipt:
        pre=json.loads(x.compare_pre_receipt.read_text())
        assert pre['phase']=='pre' and pre['source_revision']==SOURCE_M0_HEAD
        assert pre['published_fixture_sha256']==fixture_hash and pre['fixture_manifest_sha256']==FROZEN_MANIFEST_SHA256
        assert pre['versions']==versions and pre['all_expected_outputs_and_repetition_paths']
        assert pre['checked']==pre['matches']==18
        assert pre['registry_data_sha256']!=sha(repo/registry_file)
        unchanged=(rollback==pre['actual_rollback_loads'])
        comparison={'pre_receipt_sha256':sha(x.compare_pre_receipt),
                    'actual_rollback_outputs_and_weights_byte_identical':unchanged,
                    'warm_p50_ms_pre_and_candidate':{d:[pre['directions'][d]['warm_p50_ms'],timings[d]['warm_p50_ms']] for d in FIXED_SCOPE},
                    'warm_timing_note':'Descriptive warm single-request measurements; not a speedup or threshold claim.'}
        ok &= unchanged
    assert observed_sources=={f:sha(repo/f) for f in sources}, 'Verification checkout changed during replay'
    receipt={
      'source_file_sha256':observed_sources,
      'phase':x.phase,'attempt':x.attempt,'source_revision':head,'published_fixture_sha256':fixture_hash,
      'fixture_manifest_sha256':sha(x.manifest),'registry_select_implementation':registry_path,
      'pre_comparison':comparison,'pipeline_implementation_sha256':sha(repo/'src/bridgecaption/pipeline.py'),
      'translate_implementation':translate_path,'registry_select_implementation_sha256':sha(repo/registry_path),
      'registry_data_sha256':sha(repo/registry_file),
      'translation_implementation_sha256':sha(repo/translate_path),'extra_post_registry_attestation':attest,
      'versions':versions,'pre_post_scope':'historical specialist baselines, text only; no single-model candidate, audio, holdout, human adequacy, production enablement, or Mandarin inference',
      'base_pins':pins,'matches':sum(item['match'] for item in results),'checked':len(results),
      'repetition_route_matches':sum(item['repetition_retry_matched'] for item in results),
      'all_expected_outputs_and_repetition_paths':ok,'directions':timings,'results':results,'actual_rollback_loads':rollback,
      'pinned_artifact_checkout_source_revision':SOURCE_M0_HEAD,
      'unsupported_routes_explicitly_rejected':['zh-en','en-zh','cmn-en','en-cmn','en-fr','fr-es','auto-en'],
      'process_peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    assert not x.receipt.exists(),x.receipt
    x.receipt.parent.mkdir(exist_ok=True,parents=True)
    x.receipt.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(f'{x.phase} {x.attempt}: {receipt["matches"]}/{receipt["checked"]} exact; 2 actual rollbacks; registry={registry_path}; peak RSS={receipt["process_peak_rss_kib_linux"]} KiB.',flush=True)
    return 0 if ok else 1

if __name__=='__main__':
    raise SystemExit(main())
