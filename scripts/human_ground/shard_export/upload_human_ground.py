"""Export exact ground training inputs to verified 32-frame TARs over one SSH."""
from __future__ import annotations
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
import time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path('/media/robosense/DataCenterNew/Kyle/ULIP-2/outputs/ulip_human_ground_v1')
DEST='/data/timon.he/Datasets/SIM/human_ground'
REMOTE = r'''
import hashlib,json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
stream=sys.stdin.buffer
def emit(x):print(json.dumps(x),flush=True)
def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.partial');temp.write_bytes(data);os.replace(temp,path)
manifest=json.loads(stream.readline());run_id=manifest['run_id']
old=root/'job_manifest.json'
if old.exists() and json.loads(old.read_text())['run_id']!=run_id:raise ValueError('Different run exists')
if not old.exists() and any(root.iterdir()):raise ValueError('Nonempty unrecognized destination')
atomic(old,(json.dumps(manifest,indent=2)+'\n').encode())
(root/'shards').mkdir(exist_ok=True);(root/'receipts').mkdir(exist_ok=True)
done={}
for p in sorted((root/'receipts').glob('capture_*.tar.json')):
    r=json.loads(p.read_text());f=root/'shards'/r['name']
    if r['run_id']!=run_id or not f.exists() or f.stat().st_size!=r['bytes']:raise ValueError('Invalid receipt')
    done[r['name']]=r
atomic(root/'uploaded.txt',(''.join(n+'\n' for n in sorted(done))).encode())
emit({'ready':True,'receipts':done})
while True:
    line=stream.readline()
    if not line:raise EOFError('Upload interrupted; not complete')
    header=json.loads(line)
    if header.get('finish'):
        if len(done)!=manifest['expected_shards'] or sum(r['frames'] for r in done.values())!=manifest['expected_frames']:raise ValueError('Incomplete dataset')
        atomic(root/'uploaded.txt',(''.join(n+'\n' for n in sorted(done))+'done\n').encode())
        atomic(root/'progress.json',json.dumps({'status':'complete','shards':len(done),'frames':sum(r['frames'] for r in done.values()),'run_id':run_id},indent=2).encode())
        emit({'complete':True});break
    name=header['name']
    if Path(name).name!=name or not name.startswith('capture_') or not name.endswith('.tar'):raise ValueError('Unsafe name')
    target=root/'shards'/name
    if name in done:raise ValueError('Duplicate shard')
    partial=target.with_suffix('.tar.partial');h=hashlib.sha256();remaining=header['bytes']
    with partial.open('wb') as f:
        while remaining:
            block=stream.read(min(1024*1024,remaining))
            if not block:raise EOFError('Interrupted payload')
            f.write(block);h.update(block);remaining-=len(block)
        f.flush();os.fsync(f.fileno())
    if h.hexdigest()!=header['sha256']:raise ValueError('SHA256 mismatch')
    os.replace(partial,target)
    receipt=dict(header,run_id=run_id,verified=True,verified_unix=time.time())
    atomic(root/'receipts'/(name+'.json'),(json.dumps(receipt)+'\n').encode());done[name]=receipt
    with (root/'uploaded.txt').open('a') as f:f.write(name+'\n');f.flush()
    atomic(root/'progress.json',json.dumps({'status':'uploading','shards':len(done),'frames':sum(r['frames'] for r in done.values()),'expected_frames':manifest['expected_frames']},indent=2).encode())
    emit({'uploaded':name,'sha256':header['sha256']})
'''

def digest(data):return hashlib.sha256(data).hexdigest()
def encode_ply(points):
    if points.shape!=(10000,3) or not np.isfinite(points).all() or (points[:,2]<0).any() or (points[8000:,2]!=0).any():
        raise ValueError('Invalid ground input')
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1'),('instance_id','<u4')])
    a=np.zeros(10000,dtype=dtype)
    for i,k in enumerate('xyz'):a[k]=points[:,i]
    for k in ('red','green','blue'):a[k]=102
    a['instance_id'][:8000]=1
    header=('ply\nformat binary_little_endian 1.0\ncomment ground_training_export_v1 units_m Z_up floor_Z_0\n'
        'element vertex 10000\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uint instance_id\nend_header\n')
    payload=header.encode()+a.tobytes()
    decoded=np.frombuffer(payload.split(b'end_header\n',1)[1],dtype=dtype)
    if not np.array_equal(np.column_stack([decoded[k] for k in 'xyz']),points):raise ValueError('PLY roundtrip mismatch')
    return payload

def main():
    out=ROOT/'shard_export';out.mkdir(exist_ok=True)
    records=[json.loads(x) for x in (ROOT/'human_text/data/records.jsonl').read_text().splitlines()]
    splits={}
    for split in ('train','validation','test'):
        for line in (ROOT/'human_text/data'/f'{split}.jsonl').read_text().splitlines():
            sid=json.loads(line)['sample_id']
            if sid in splits:raise ValueError('Split overlap')
            splits[sid]=split
    if set(splits)!={r['sample_id'] for r in records}:raise ValueError('Split coverage mismatch')
    chunks=sorted((ROOT/'human_text/input_chunks').glob('chunk_*.npz'))
    if len({r['sample_id'] for r in records})!=len(records):raise ValueError('Duplicate sample IDs')
    if len(chunks)!=(len(records)+31)//32:raise ValueError('Chunk count mismatch')
    identity=json.loads((ROOT/'human_text/features/identity.json').read_text())
    manifest=dict(schema='human-ground-frame-tar-v1',source=str(ROOT),source_identity=identity,
        expected_frames=len(records),expected_shards=len(chunks),frames_per_shard=32,
        labels=dict(Counter(r['label'] for r in records)),splits=dict(Counter(splits.values())),
        pointcloud=dict(points=10000,human_points=8000,ground_points=2000,units='m',
            coordinates='source lidar-heading ground-origin XYZ; Z-up; floor Z=0; not sensor frame; not normalized',
            disk_radius_m=1.5,human_translation_m=[0,0,-0.1],negative_z_points_removed=True,
            human_sampling='8000 with replacement if source has fewer points',rgb=[102,102,102],
            instance_id={'0':'synthetic ground','1':'human'}),
        joints_included=False,joints_reason='Cache contains no consistently transformed skeleton; original joints not copied',
        files_per_frame=['lidar/lidar_<source_index>.ply','label.txt','frame_manifest.json'],
        local_source_deleted=False)
    manifest['run_id']=digest(json.dumps(manifest,sort_keys=True).encode())
    (out/'job_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','openpai@10.199.2.164',
        'python3 -u -c '+shlex.quote(REMOTE)+' '+shlex.quote(DEST)],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    def send(x):proc.stdin.write((json.dumps(x)+'\n').encode());proc.stdin.flush()
    def read():
        line=proc.stdout.readline()
        if not line:raise RuntimeError('Receiver exited: '+str(proc.poll()))
        return json.loads(line)
    send(manifest);ready=read();done=ready['receipts'];started=time.monotonic();uploaded_bytes=0
    print('Started',len(records),'frames',len(chunks),'shards; already uploaded',len(done),flush=True)
    for ci,path in enumerate(chunks):
        name=f'capture_{ci:06d}.tar';meta=json.loads(path.with_suffix('.json').read_text())
        batch=records[ci*32:(ci+1)*32]
        if meta['record_ids']!=[r['sample_id'] for r in batch]:raise ValueError('Sample ordering mismatch')
        if name in done:
            if done[name]['source_chunk_sha256']!=meta['sha256']:raise ValueError('Resume source mismatch')
            continue
        data=path.read_bytes()
        if digest(data)!=meta['sha256']:raise ValueError('Source chunk checksum mismatch')
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:xyz=archive['input_xyz_m']
        if xyz.shape!=(len(batch),10000,3):raise ValueError('Unexpected array shape')
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w',format=tarfile.PAX_FORMAT) as tar:
            for i,r in enumerate(batch):
                fid=r['frame_id'];lidar_name=Path(r['ply_member']).name
                if '/' in fid or fid in ('.','..'):raise ValueError('Unsafe frame ID')
                fm=dict(schema='human-ground-frame-v1',frame_id=fid,sample_id=r['sample_id'],label=r['label'],
                    validation_split=splits[r['sample_id']],sequence_id=r['sequence_id'],source_record=r,
                    preprocessing=meta['inputs'][i],pointcloud=manifest['pointcloud'],source_chunk=path.name,
                    source_chunk_sha256=meta['sha256'],generation_method='cached_pointcloud_augmentation',
                    joints_included=False,run_id=manifest['run_id'])
                files={f'lidar/{lidar_name}':encode_ply(xyz[i]),'label.txt':(r['label']+'\n').encode(),
                    'frame_manifest.json':(json.dumps(fm,indent=2)+'\n').encode()}
                for relative,blob in files.items():
                    info=tarfile.TarInfo(fid+'/'+relative);info.size=len(blob);info.mode=0o644
                    tar.addfile(info,io.BytesIO(blob))
        payload=buffer.getvalue();sha=digest(payload)
        send(dict(name=name,bytes=len(payload),sha256=sha,frames=len(batch),frame_ids=[r['frame_id'] for r in batch],source_chunk_sha256=meta['sha256']))
        proc.stdin.write(payload);proc.stdin.flush();ack=read()
        if ack.get('uploaded')!=name or ack.get('sha256')!=sha:raise ValueError('Bad upload acknowledgement')
        uploaded_bytes+=len(payload)
        progress=dict(status='uploading',shards=ci+1,expected_shards=len(chunks),frames=min((ci+1)*32,len(records)),expected_frames=len(records),wall_s=time.monotonic()-started,uploaded_bytes=uploaded_bytes,ssh_connections=1)
        (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
        if ci%25==0 or ci==len(chunks)-1:print(json.dumps(progress),flush=True)
    send({'finish':True});ack=read()
    if not ack.get('complete'):raise ValueError('Completion not acknowledged')
    proc.stdin.close()
    if proc.wait()!=0:raise ValueError('SSH failed')
    summary=dict(status='complete',frames=len(records),shards=len(chunks),wall_s=time.monotonic()-started,uploaded_bytes=uploaded_bytes,remote=DEST,ssh_connections=1)
    (out/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
