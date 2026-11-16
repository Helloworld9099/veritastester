"""Registered continued neural training. No app/provider changes or integration."""
import os,sys,json,random,time,math,hashlib,statistics,resource,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
sys.path[:0]=[str(BASE/'translation_training_20261116/deps'),str(BASE/'veritastester-localmt-deps'),str(BASE/'veritastester-validation-deps')]
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import torch
from torch import nn
from safetensors.torch import save_file,load_file
from transformers import MarianTokenizer,MarianMTModel
from sacrebleu.metrics import CHRF,BLEU
SEED=202611162
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED);random.seed(SEED)
class LoRALinear(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base;self.scale=2.;self.lora_A=nn.Parameter(torch.empty(8,base.in_features));nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5));self.lora_B=nn.Parameter(torch.zeros(base.out_features,8))
 def forward(self,x):return self.base(x)+((x@self.lora_A.T)@self.lora_B.T)*self.scale

def rows(p):return[json.loads(x)for x in p.read_text().splitlines()]
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def main(lang):
 started=time.monotonic();p=ROOT/lang;old=BASE/('translation_fr_training_20261116'if lang=='french'else 'translation_training_20261116');lr=1e-4 if lang=='french'else 5e-5;planned=256 if lang=='french'else 128
 tr,dev,test,guard=[rows(p/(s+'.jsonl'))for s in['train','dev','test','guard']]
 norm=lambda s:' '.join(s.casefold().split())
 for a,b in[(tr,dev),(tr,test),(dev,test)]:
  for key in['source','reference','group']:
   assert not{norm(r[key])if key!='group'else r[key]for r in a}&{norm(r[key])if key!='group'else r[key]for r in b}
 assert not{norm(r['source'])for r in tr+dev+test}&{norm(r['source'])for r in guard}
 modelpath=old/'model';warmfile=old/'adapter_selected.safetensors';protocol={'language':lang,'seed':SEED,'model':json.loads((modelpath/'repository_info.json').read_text()),'original_checkpoint_sha256':hashlib.sha256(warmfile.read_bytes()).hexdigest(),'train_rows':len(tr),'dev_rows':len(dev),'test_rows':len(test),'guard_rows':len(guard),'data_manifest_sha256':hashlib.sha256((p/'data_manifest.json').read_bytes()).hexdigest(),'recipe':{'optimizer':'AdamW','lr':lr,'weight_decay':.01,'microbatch':4,'gradient_accumulation':4,'effective_batch':16,'planned_updates':planned,'grad_clip':1.,'max_source_target_tokens':96,'lora_rank':8,'lora_alpha':16,'cpu_threads':4,'pure_training_budget_seconds':600},'warm_start':lang=='french','warm_start_detail':'Previous adapter parameters only, fresh optimizer; NOT a resumed optimizer state'if lang=='french'else'Zero-initialized adapters on unchanged base; prior adapter comparator only','selection':'Highest NEW dev corpus chrF, tie favors earliest candidate. French candidates: base, previous adapter, continued step128/256. Spanish candidates: base, new64/128; prior adapter evaluated as control, ineligible for selection. Freeze before opening test outputs. No test-based choice.','decoding':dict(beam_size=4,batch=8,max_source_tokens=128,max_generated_tokens=96),'warning':'Context-free text translation only. New project-held-out examples, not guaranteed unknown to pretrained OPUS. Prior tests and parent synthetic audit are consumed and not used for selection. No app integration, activation, paid API, push or deployment.'}
 write(p/'protocol.json',protocol);print('REGISTERED',lang,planned,'updates',flush=True)
 tok=MarianTokenizer.from_pretrained(str(modelpath),local_files_only=True);model=MarianMTModel.from_pretrained(str(modelpath),local_files_only=True,weights_only=True)
 for t in model.parameters():t.requires_grad_(False)
 targets=[]
 for name,m in list(model.named_modules()):
  if isinstance(m,nn.Linear)and name.endswith(('q_proj','v_proj')):
   parent,attr=name.rsplit('.',1);setattr(model.get_submodule(parent),attr,LoRALinear(m));targets.append(name)
 params=[x for x in model.parameters()if x.requires_grad];protocol['trainable_parameters']=sum(x.numel()for x in params);protocol['target_modules']=targets;write(p/'protocol.json',protocol);model.config.use_cache=False
 def state():return{k:v.detach().cpu().clone()for k,v in model.state_dict().items()if'.lora_'in k}
 def load(s):
  assert set(s)=={k for k in model.state_dict()if'.lora_'in k}
  with torch.no_grad():
   for k,v in s.items():assert model.state_dict()[k].shape==v.shape;model.state_dict()[k].copy_(v)
 def batch(rs):
  b=tok([r['source']for r in rs],text_target=[r['reference']for r in rs],padding=True,truncation=True,max_length=96,return_tensors='pt');b['labels'][b['labels']==tok.pad_token_id]=-100;return b
 def translate(rs,tag):
  model.eval();out=[];elapsed=[]
  with torch.no_grad():
   for i in range(0,len(rs),8):
    b=tok([r['source']for r in rs[i:i+8]],padding=True,truncation=True,max_length=128,return_tensors='pt');t=time.monotonic();z=model.generate(**b,num_beams=4,max_new_tokens=96,use_cache=True);elapsed.append((time.monotonic()-t)/len(z));out+=tok.batch_decode(z,skip_special_tokens=True)
    if (i+len(z))%128==0:print('DECODE',tag,i+len(z),'/',len(rs),flush=True)
  return out,statistics.mean(elapsed)
 def metric(pred,rs):
  refs=[r['reference']for r in rs];c=CHRF();b=BLEU();return{'n':len(rs),'chrf':c.corpus_score(pred,[refs]).score,'bleu':b.corpus_score(pred,[refs]).score,'chrf_signature':str(c.get_signature()),'bleu_signature':str(b.get_signature())}
 base=state();previous=load_file(str(warmfile));history=[];candidates={};preds_dev={}
 for name,st in[('base',base),('previous',previous)]:
  load(st);pred,_=translate(dev,'dev-'+name);m=metric(pred,dev);history.append(dict(candidate=name,metrics=m));preds_dev[name]=pred
  if name=='base'or lang=='french':candidates[name]=(m,st)
  print('DEV',name,m,flush=True)
 write(p/'dev_history.json',history)
 load(previous if lang=='french'else base);save_file(state(),str(p/'adapter_start.safetensors'))
 opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01);opt.zero_grad(set_to_none=True);order=list(range(len(tr)));random.Random(SEED).shuffle(order);micro=step=0;train_compute=0.;losses=[]
 while step<planned and train_compute<600:
  t=time.monotonic();ids=order[(micro*4)%len(order):(micro*4)%len(order)+4];assert len(ids)==4
  model.train();loss=model(**batch([tr[i]for i in ids])).loss;assert torch.isfinite(loss);(loss/4).backward();micro+=1
  if micro%4==0:
   gn=float(torch.nn.utils.clip_grad_norm_(params,1.));opt.step();opt.zero_grad(set_to_none=True);step+=1
   losses.append(dict(step=step,loss=float(loss.detach()),grad_norm=gn,train_compute_seconds=train_compute+time.monotonic()-t))
  train_compute+=time.monotonic()-t
  if micro%4==0 and step%32==0:print('TRAIN',lang,step,'loss',round(float(loss.detach()),4),'compute',round(train_compute,1),flush=True)
  checkpoints=[128,256]if lang=='french'else[64,128]
  if micro%4==0 and step in checkpoints:
   st=state();save_file(st,str(p/f'adapter_step{step}.safetensors'));pred,_=translate(dev,'dev-step'+str(step));m=metric(pred,dev);name='step'+str(step);candidates[name]=(m,st);history.append(dict(candidate=name,metrics=m));preds_dev[name]=pred;write(p/'dev_history.json',history);print('DEV',name,m,flush=True)
 save_file(state(),str(p/'adapter_last.safetensors'));write(p/'training_log.json',losses)
 if step not in checkpoints and step:
  st=state();pred,_=translate(dev,'dev-budget-stop');m=metric(pred,dev);candidates['budget_stop'+str(step)]=(m,st);history.append(dict(candidate='budget_stop'+str(step),metrics=m));write(p/'dev_history.json',history)
 selected=max(candidates,key=lambda k:candidates[k][0]['chrf']);best,chosen=candidates[selected];save_file(chosen,str(p/'adapter_selected.safetensors'));write(p/'selected.json',dict(candidate=selected,dev=best,completed_updates=step,planned_updates=planned,selection_frozen_before_test=True));print('FROZEN',lang,selected,flush=True)
 report={'protocol':protocol,'completed_updates':step,'pure_training_seconds':train_compute,'selected':selected,'dev_history':history,'evaluations':{}}
 for name,rs in[('new_source_test',test),('new_external_guard',guard)]:
  outputs={};metrics={};states={'base':base,'previous':previous,'selected':chosen};seen={}
  for arm,st in states.items():
   key=hashlib.sha256(b''.join(v.numpy().tobytes()for k,v in sorted(st.items()))).hexdigest()
   if key in seen:out,secs=seen[key]
   else:load(st);out,secs=translate(rs,name+'-'+arm);seen[key]=(out,secs)
   outputs[arm]=out;metrics[arm]=dict(**metric(out,rs),batch_amortized_seconds_per_sentence=secs)
  report['evaluations'][name]=metrics;(p/(name+'_outputs.jsonl')).write_text(''.join(json.dumps(dict(**r,**{arm:outs[i]for arm,outs in outputs.items()}),ensure_ascii=False)+'\n'for i,r in enumerate(rs)));write(p/'report_partial.json',report)
  print('TEST',lang,name,metrics,flush=True)
 # Validate selected serialization and no nonfinite adapter tensors.
 load(chosen);model.eval();b=batch(dev[:4])
 with torch.no_grad():before=model(**b).logits
 load(base);load(load_file(str(p/'adapter_selected.safetensors')))
 with torch.no_grad():after=model(**b).logits
 assert torch.equal(before,after);assert all(torch.isfinite(v).all()for v in chosen.values());report['selected_replay_exact']=True;report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;report['elapsed_seconds']=time.monotonic()-started;write(p/'report.json',report);print('COMPLETE',lang,selected,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--language',choices=['french','spanish'],required=True);main(a.parse_args().language)
