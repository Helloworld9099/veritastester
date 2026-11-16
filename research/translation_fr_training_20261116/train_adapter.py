"""Free CPU Marian LoRA pilot; frozen backbone; isolated research only."""
import os,sys,json,random,time,math,hashlib,statistics,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT.parent/'translation_training_20261116/deps'),str(ROOT.parent/'veritastester-localmt-deps'),str(ROOT.parent/'veritastester-validation-deps')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from torch import nn
from transformers import MarianTokenizer,MarianMTModel
from safetensors.torch import save_file,load_file
from sacrebleu.metrics import CHRF,BLEU
SEED=20261116
torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(SEED);random.seed(SEED)

def read(name):return [json.loads(x)for x in (ROOT/(name+'.jsonl')).read_text().splitlines()]
def write(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))

class LoRALinear(nn.Module):
 def __init__(self,base,rank=8,alpha=16):
  super().__init__();self.base=base;self.scaling=alpha/rank;self.enabled=True
  self.lora_A=nn.Parameter(torch.empty(rank,base.in_features));nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
  self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank))
 def forward(self,x):
  y=self.base(x)
  return y+((x@self.lora_A.T)@self.lora_B.T)*self.scaling if self.enabled else y

def main():
 start=time.monotonic();train,dev,test=read('train'),read('dev'),read('test')
 for a,b in [(train,dev),(train,test),(dev,test)]:
  assert not{r['block']for r in a}&{r['block']for r in b}
  assert not{' '.join(r['source'].casefold().split())for r in a}&{' '.join(r['source'].casefold().split())for r in b}
 tokenizer=MarianTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
 t=time.monotonic();model=MarianMTModel.from_pretrained(str(ROOT/'model'),local_files_only=True,weights_only=True)
 load_seconds=time.monotonic()-t
 for p in model.parameters():p.requires_grad_(False)
 modules=[]
 for name,child in list(model.named_modules()):
  if isinstance(child,nn.Linear)and name.endswith(('q_proj','v_proj')):
   parent_name,attr=name.rsplit('.',1);parent=model.get_submodule(parent_name);adapter=LoRALinear(child);setattr(parent,attr,adapter);modules.append(name)
 parameters=[p for p in model.parameters()if p.requires_grad];trainable=sum(p.numel()for p in parameters)
 model.config.use_cache=False
 def enabled(value):
  for m in model.modules():
   if isinstance(m,LoRALinear):m.enabled=value
 def state():return{k:v.detach().cpu().clone()for k,v in model.state_dict().items()if '.lora_'in k}
 def load(state):
  with torch.no_grad():
   for k,v in state.items():model.state_dict()[k].copy_(v)
 def batch(rows):
  data=tokenizer([r['source']for r in rows],text_target=[r['reference']for r in rows],return_tensors='pt',padding=True,truncation=True,max_length=96)
  data['labels'][data['labels']==tokenizer.pad_token_id]=-100
  return data
 def nll(rows):
  model.eval();num=den=0
  with torch.no_grad():
   for i in range(0,len(rows),8):
    b=batch(rows[i:i+8]);n=int((b['labels']!=-100).sum());num+=float(model(**b).loss)*n;den+=n
  return num/den
 def translate(rows):
  model.eval();outs=[];times=[]
  with torch.no_grad():
   for i in range(0,len(rows),8):
    b=tokenizer([r['source']for r in rows[i:i+8]],return_tensors='pt',padding=True,truncation=True,max_length=128)
    t=time.monotonic();out=model.generate(**b,num_beams=4,max_new_tokens=96,use_cache=True)
    times.append((time.monotonic()-t)/len(out));outs+=tokenizer.batch_decode(out,skip_special_tokens=True)
  return outs,times
 def metric(pred,rows):
  refs=[r['reference']for r in rows];ch=CHRF();bl=BLEU()
  c,b=ch.corpus_score(pred,[refs]),bl.corpus_score(pred,[refs])
  return dict(n=len(rows),chrf=c.score,bleu=b.score,chrf_signature=str(ch.get_signature()),bleu_signature=str(bl.get_signature()))

 protocol={'task':'Genuine French-English neural translation LoRA fine-tuning; not contextual translation/ASR','seed':SEED,'model':json.loads((ROOT/'model/repository_info.json').read_text()),'torch':torch.__version__,'device':'CPU, 4 threads','frozen_backbone':True,'trainable_parameters':trainable,'total_parameters':sum(p.numel()for p in model.parameters()),'lora':dict(rank=8,alpha=16,target_modules=modules),'train_rows':len(train),'dev_rows':len(dev),'test_rows':len(test),'batch_size':4,'gradient_accumulation':2,'optimizer':'AdamW lr=2e-4 weight_decay=.01','planned_optimizer_steps':128,'budget_seconds':600,'dev_selection':'baseline, step64 and step128; select greatest dev chrF; baseline remains eligible; NLL descriptive only','training_max_tokens':96,'decoding':'beam4, max_new_tokens96, batch8','warnings':['Connected-translation-component disjoint; author/topic/template independence not guaranteed.','Base OPUS may have seen Tatoeba or DiaBLa in pretraining; unknown.','DiaBLa guard excludes all 40 smoke-dialogue groups and is evaluated only after selection.','Only French-English text; no ASR or multilingual/slang-accuracy claim.']}
 # External dialogue guard frozen by prepare_from_handoff.py, excluded from training.
 wmt=read('diabla_guard')
 write('protocol.json',protocol);print('LOADED',load_seconds,'trainable',trainable,flush=True)
 baseline_state=state();save_file(baseline_state,str(ROOT/'adapter_initial.safetensors'))
 # Initial zero-output adapters must preserve the frozen model exactly.
 b=batch(dev[:4]);model.eval()
 with torch.no_grad():
  enabled(False);base_logits=model(**b).logits;enabled(True);adapt_logits=model(**b).logits
 assert torch.equal(base_logits,adapt_logits)
 base_nll=nll(dev);base_dev,times=translate(dev);best=metric(base_dev,dev);best_step=0;best_state=baseline_state
 dev_history=[dict(step=0,dev_nll=base_nll,metrics=best)];write('dev_history.json',dev_history)
 print('BASELINE DEV',base_nll,best,flush=True)
 optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=.01)
 rng=random.Random(SEED);order=list(range(len(train)));rng.shuffle(order);micro=0;step=0;loss_log=[];train_start=time.monotonic();optimizer.zero_grad(set_to_none=True)
 while step<128 and time.monotonic()-train_start<600:
  ids=order[(micro*4)%len(order):(micro*4)%len(order)+4]
  if len(ids)<4:ids+=order[:4-len(ids)]
  model.train();loss=model(**batch([train[i]for i in ids])).loss
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; no candidate promotion')
  (loss/2).backward();micro+=1
  if micro%2==0:
   grad=float(torch.nn.utils.clip_grad_norm_(parameters,1.));optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
   loss_log.append(dict(step=step,loss=float(loss.detach()),gradient_norm=grad,seconds=time.monotonic()-train_start))
   if step%16==0:print('STEP',step,'loss',round(float(loss.detach()),4),'elapsed',round(time.monotonic()-train_start,1),flush=True)
   if step in [64,128]:
    d_nll=nll(dev);p,_=translate(dev);m=metric(p,dev);dev_history.append(dict(step=step,dev_nll=d_nll,metrics=m));write('dev_history.json',dev_history);save_file(state(),str(ROOT/f'adapter_step{step}.safetensors'))
    if m['chrf']>best['chrf']:best,best_step,best_state=m,step,state()
    print('DEV',step,d_nll,m,'selected',best_step,flush=True)
 # Preserve an incomplete pilot, but never pretend it completed planned training.
 save_file(state(),str(ROOT/'adapter_last.safetensors'));write('training_log.json',loss_log)
 if step not in [64,128]and step:
  d_nll=nll(dev);p,_=translate(dev);m=metric(p,dev);dev_history.append(dict(step=step,dev_nll=d_nll,metrics=m,budget_stop=True))
  if m['chrf']>best['chrf']:best,best_step,best_state=m,step,state()
 write('dev_history.json',dev_history);save_file(best_state,str(ROOT/'adapter_selected.safetensors'));write('selected.json',dict(step=best_step,dev=best,trained_steps=step,selection_frozen_before_test=True))
 report=dict(protocol=protocol,model_load_seconds=load_seconds,steps_completed=step,train_seconds=time.monotonic()-train_start,selected_step=best_step,initial_adapter_exact_backbone_logits=True,dev_history=dev_history,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 for name,rows in [('tatoeba_test',test),('diabla_guard',wmt)]:
  enabled(False);base,bt=translate(rows);enabled(True);load(best_state);pred,pt=translate(rows)
  report[name]={'baseline':metric(base,rows),'selected':metric(pred,rows),'baseline_seconds_per_sentence':statistics.mean(bt),'selected_seconds_per_sentence':statistics.mean(pt)}
  (ROOT/(name+'_outputs.jsonl')).write_text(''.join(json.dumps(dict(**r,baseline=a,selected=b),ensure_ascii=False)+'\n'for r,a,b in zip(rows,base,pred)))
 # Demonstrate that the selected adapter checkpoint can be restored safely.
 load(baseline_state);load(load_file(str(ROOT/'adapter_selected.safetensors')))
 with torch.no_grad():reloaded=model(**b).logits
 load(best_state)
 with torch.no_grad():expected=model(**b).logits
 assert torch.equal(reloaded,expected);report['selected_adapter_replay_exact']=True
 report['total_seconds']=time.monotonic()-start;report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 write('report.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
