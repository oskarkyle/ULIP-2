import json,subprocess,hashlib,time,shutil,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
base=Path('/media/robosense/DataCenterNew/Kyle/ULIP-2/checkpoints')
x=next(x for x in json.load(open('/media/robosense/DataCenterNew/Kyle/ULIP-2/setup/ulip-openclip-files.json')) if x['path']=='open_clip_pytorch_model.bin')
dest=base/x['path']; part=base/(x['path']+'.part'); chunks=base/'openclip_chunks'; chunks.mkdir(exist_ok=True)
manifest=chunks/'ranges.json'
if not manifest.exists():
 start=part.stat().st_size if part.exists() else 0
 ranges=[(a,min(a+256*1024*1024,x['size'])-1) for a in range(start,x['size'],256*1024*1024)]
 manifest.write_text(json.dumps({'prefix':start,'ranges':ranges}))
plan=json.loads(manifest.read_text())
def get(pair):
 a,b=pair; output=chunks/f'{a}-{b}.bin'; tmp=output.with_suffix('.part')
 if a==9932644706:
  while not output.exists(): time.sleep(2)
  assert output.stat().st_size==b-a+1
  return output
 if output.exists() and output.stat().st_size==b-a+1: return output
 for attempt in range(30):
  offset=tmp.stat().st_size if tmp.exists() else 0
  if offset==b-a+1:
   tmp.rename(output);print('CHUNK',a,b,flush=True);return output
  url='https://hf-mirror.com/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/'+x['path']+f'?download=true&part={a}&offset={offset}'
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={a+offset}-{b}'})
   with urllib.request.urlopen(req,timeout=40) as r:
    assert r.status==206,r.status
    assert r.headers['Content-Range'].startswith(f'bytes {a+offset}-'),r.headers['Content-Range']
    with tmp.open('ab') as f:
     while True:
      block=r.read(256*1024)
      if not block:break
      f.write(block)
   assert tmp.stat().st_size==b-a+1,tmp.stat().st_size
   tmp.rename(output);print('CHUNK',a,b,flush=True);return output
  except Exception as e: print('RETRY',a,type(e).__name__,flush=True);time.sleep(2)
 raise RuntimeError(f'Chunk failed {a}-{b}')
if not dest.exists():
 with ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(get,plan['ranges']))
 with part.open('r+b') as f:
  f.truncate(plan['prefix']);f.seek(plan['prefix'])
  for path in results:
   with path.open('rb') as r: shutil.copyfileobj(r,f,8*1024*1024)
 assert part.stat().st_size==x['size']
 h=hashlib.sha256()
 with part.open('rb') as f:
  for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
 assert h.hexdigest()==x['lfs']['oid'],h.hexdigest()
 part.rename(dest)
 print('VERIFIED',dest,h.hexdigest(),flush=True)
 for path in results:path.unlink()
