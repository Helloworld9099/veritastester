"""Validate public corpus snapshots; download only with explicit opt-in.

No credentials, package installs, model inference, training or publication.
Source hashes pin snapshot bytes. No archives are extracted by this script.
"""
from __future__ import annotations
import argparse,hashlib,json,os,tempfile,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def check(path,source):
 return path.is_file() and path.stat().st_size==source['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--raw-dir',type=Path,default=ROOT/'raw')
 p.add_argument('--download',action='store_true')
 p.add_argument('--accept-source-licenses',action='store_true')
 a=p.parse_args()
 if a.download and not a.accept_source_licenses:p.error('Read README license/provenance notices, then opt in with --accept-source-licenses')
 sources=json.loads((ROOT/'sources.json').read_text())
 failures=[]
 for s in sources:
  target=a.raw_dir/s['file']
  if check(target,s):print('VERIFIED',s['file']);continue
  if not a.download:
   failures.append(s['file']);print('MISSING_OR_WRONG_HASH',s['file']);continue
  if target.exists():p.error('Existing bytes differ: refusing to overwrite '+str(target))
  a.raw_dir.mkdir(parents=True,exist_ok=True);os.chmod(a.raw_dir,0o700)
  fd,tmp=tempfile.mkstemp(prefix=s['file']+'.',suffix='.part',dir=a.raw_dir)
  try:
   with os.fdopen(fd,'wb') as out,urllib.request.urlopen(s['url'],timeout=60) as incoming:
    total=0
    while True:
     block=incoming.read(1024*1024)
     if not block:break
     total+=len(block)
     if total>s['bytes']:raise ValueError('Source exceeds pinned byte count')
     out.write(block)
   if not check(Path(tmp),s):raise ValueError('Source size/hash differs from pinned snapshot')
   os.replace(tmp,target);os.chmod(target,0o600);print('DOWNLOADED_AND_VERIFIED',s['file'])
  finally:
   if Path(tmp).exists():Path(tmp).unlink()
 if failures:raise SystemExit('Validation only; missing/wrong files: '+', '.join(failures))
if __name__=='__main__':main()
