"""Local research inference only; no app integration or network requests."""
import argparse,json
from pathlib import Path
import train_adapter as t
from safetensors.torch import load_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--text',required=True);p.add_argument('--baseline',action='store_true');a=p.parse_args()
 if not a.text.strip():p.error('Nonempty text required')
 tokenizer=t.MarianTokenizer.from_pretrained(str(t.ROOT/'model'),local_files_only=True)
 model=t.MarianMTModel.from_pretrained(str(t.ROOT/'model'),local_files_only=True,weights_only=True)
 protocol=json.loads((t.ROOT/'protocol.json').read_text());expected=protocol['lora']['target_modules'];found=[]
 for name,child in list(model.named_modules()):
  if name in expected:
   parent_name,attr=name.rsplit('.',1);setattr(model.get_submodule(parent_name),attr,t.LoRALinear(child));found.append(name)
 assert sorted(found)==sorted(expected)
 adapter=load_file(str(t.ROOT/'adapter_selected.safetensors'))
 names={k for k in model.state_dict()if '.lora_'in k};assert names==set(adapter)
 with t.torch.no_grad():
  for k,v in adapter.items():
   assert model.state_dict()[k].shape==v.shape
   model.state_dict()[k].copy_(v)
 for m in model.modules():
  if isinstance(m,t.LoRALinear):m.enabled=not a.baseline
 model.eval()
 inputs=tokenizer(a.text,return_tensors='pt',truncation=True,max_length=128)
 with t.torch.no_grad():output=model.generate(**inputs,num_beams=4,max_new_tokens=96)
 print(json.dumps({'output':tokenizer.decode(output[0],skip_special_tokens=True),'candidate':not a.baseline,'warning':'Research-only context-free translation; not independently certified or integrated.'},ensure_ascii=False))
if __name__=='__main__':main()
