"""Independently load saved LoRA adapters and replay fixed output prefixes."""
from pathlib import Path
import importlib.util,json,sys,time
ROOT=Path('/workspace/scratch/2b30092347a7');OUT=Path(__file__).resolve().parent
report={}
for lang,p,sets in [('es',ROOT/'translation_training_20261116',['europarl_test','wmt_guard']),('fr',ROOT/'translation_fr_training_20261116',['tatoeba_test','diabla_guard'])]:
 if len(sys.argv)>1 and lang!=sys.argv[1]:continue
 spec=importlib.util.spec_from_file_location('pilot_'+lang,p/'train_adapter.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 from safetensors.torch import load_file
 torch=t.torch
 tokenizer=t.MarianTokenizer.from_pretrained(str(p/'model'),local_files_only=True)
 model=t.MarianMTModel.from_pretrained(str(p/'model'),local_files_only=True,weights_only=True)
 before=[x.requires_grad for x in model.parameters()]
 expected=json.loads((p/'protocol.json').read_text())['lora']['target_modules'];found=[]
 for name,child in list(model.named_modules()):
  if name in expected:
   parent,attr=name.rsplit('.',1);setattr(model.get_submodule(parent),attr,t.LoRALinear(child));found.append(name)
 assert sorted(expected)==sorted(found)
 def enable(v):
  for m in model.modules():
   if isinstance(m,t.LoRALinear):m.enabled=v
 model.eval()
 inp=tokenizer(['Hola, buenos días.'if lang=='es'else'Bonjour, comment allez-vous?'],return_tensors='pt')
 with torch.no_grad():
  enable(False);a=model(**inp,decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]])).logits
  enable(True);b=model(**inp,decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]])).logits
 assert torch.equal(a,b)
 initial=load_file(str(p/'adapter_initial.safetensors'));selected=load_file(str(p/'adapter_selected.safetensors'))
 state=model.state_dict();keys={k for k in state if '.lora_'in k};assert keys==set(selected)==set(initial)
 with torch.no_grad():
  for k,v in selected.items():assert v.shape==state[k].shape and torch.isfinite(v).all();state[k].copy_(v)
 data={'adapter_tensor_count':len(keys),'adapter_parameters':sum(x.numel()for x in selected.values()),'nonzero_B_tensors':sum(bool(v.abs().sum()>0)for k,v in selected.items()if '.lora_B'in k),'initial_zero_output_matches_base':True,'replays':{}}
 for name in sets:
  rows=[json.loads(s)for s in(p/(name+'_outputs.jsonl')).read_text().splitlines()][:8]
  inputs=tokenizer([x['source']for x in rows],padding=True,truncation=True,max_length=128,return_tensors='pt')
  item={}
  for arm in ['baseline','selected']:
   enable(arm=='selected');start=time.monotonic()
   with torch.no_grad():generated=model.generate(**inputs,num_beams=4,max_new_tokens=96,use_cache=True)
   outputs=tokenizer.batch_decode(generated,skip_special_tokens=True)
   mismatches=[i for i,(r,x)in enumerate(zip(rows,outputs))if r[arm]!=x]
   item[arm]={'n':len(rows),'exact_output_matches':len(rows)-len(mismatches),'mismatch_indices':mismatches,'elapsed_seconds':time.monotonic()-start}
  data['replays'][name]=item
 # Count truncation without changing data or decoding.
 data['source_truncation_counts']={}
 for split in ['train','dev','test']:
  rows=[json.loads(s)for s in(p/(split+'.jsonl')).read_text().splitlines()]
  lengths=[len(tokenizer.encode(r['source']))for r in rows]
  refs=[len(tokenizer(text_target=r['reference'])['input_ids'])for r in rows]
  data['source_truncation_counts'][split]={'source_over_training96':sum(x>96 for x in lengths),'source_over_eval128':sum(x>128 for x in lengths),'reference_over_training96':sum(x>96 for x in refs)}
 report[lang]=data
 del model,tokenizer
(OUT/('checkpoint_replay_'+sys.argv[1]+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
