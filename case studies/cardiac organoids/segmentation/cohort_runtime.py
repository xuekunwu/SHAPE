"""Atomic, attested cohort journals and a single isolated GPU worker (stdlib only)."""
from __future__ import annotations
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
from uuid import uuid4


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    with temp.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class RunStore:
    """One writer; source/data/checkpoint/environment/budget identity cannot change on resume."""
    def __init__(self,root,identity,*,resume=False):
        self.root=Path(root)
        self.identity=identity
        existed=self.root.exists()
        if existed and not resume:raise ValueError('Existing run requires explicit --resume')
        if resume and not existed:raise ValueError('Resume run does not exist')
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/'.writer.lock').open('a+b')
        self.lock.seek(0);self.lock.write(b'0');self.lock.flush();self.lock.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,IOError):
            self.lock.close();raise ValueError('Run already has an active writer')
        path=self.root/'identity.json'
        if existed:
            if not path.exists() or json.loads(path.read_text())!=identity:
                self.close();raise ValueError('Resume identity mismatch')
        else:atomic_json(path,identity)
        self.identity_sha=digest(identity)

    def close(self):
        if not self.lock.closed:
            if os.name=='nt':
                import msvcrt
                self.lock.seek(0);msvcrt.locking(self.lock.fileno(),msvcrt.LK_UNLCK,1)
            self.lock.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def key(self,kind,scope):return kind+'/'+digest(scope)

    def load(self,key):
        path=self.root/key/'receipt.json'
        if not path.exists():return None
        receipt=json.loads(path.read_text())
        payload=receipt['payload']
        if receipt['identity_sha']!=self.identity_sha or receipt['payload_sha']!=digest(payload):
            raise ValueError('Corrupt or foreign journal receipt')
        for relative,sha in receipt['artifacts'].items():
            p=(self.root/relative).resolve()
            if not p.is_relative_to(self.root.resolve()) or not p.is_file() or file_sha(p)!=sha:
                raise ValueError('Changed or missing journal artifact')
        return payload

    def save(self,key,payload,artifacts=()):
        attested={str(Path(p).resolve().relative_to(self.root.resolve())):file_sha(p) for p in artifacts}
        atomic_json(self.root/key/'receipt.json',dict(identity_sha=self.identity_sha,payload=payload,payload_sha=digest(payload),artifacts=attested))


class IsolatedWorker:
    """One GPU model process. A hung request is stopped before another worker starts.

    The parent never initializes CUDA. A process failure affects only its current pass;
    every terminal failure is journaled and is not silently retried on resume.
    """
    def __init__(self,target,config,*,startup_seconds=180):
        self.target,self.config,self.startup_seconds=target,config,startup_seconds
        self.process=self.conn=None
        self.starts=0

    def start(self):
        ctx=multiprocessing.get_context('spawn')
        self.conn,child=ctx.Pipe()
        self.process=ctx.Process(target=self.target,args=(child,self.config))
        self.process.start();child.close();self.starts+=1
        if not self.conn.poll(self.startup_seconds):
            self.stop();raise RuntimeError('GPU worker startup timed out')
        try:message=self.conn.recv()
        except EOFError:
            self.stop();raise RuntimeError('GPU worker exited during startup')
        if message.get('status')!='READY':
            self.stop();raise RuntimeError('GPU worker initialization failed: '+str(message.get('error')))
        return message

    def request(self,payload,*,seconds):
        if self.process is None or not self.process.is_alive():self.start()
        started=time.monotonic()
        try:
            self.conn.send(payload)
            if self.conn.poll(seconds):
                result=self.conn.recv()
                if result.get('fatal'):
                    self.stop();raise RuntimeError('Fatal worker configuration/API error: '+str(result.get('error')))
                return result
            self.stop()
            return dict(status='TIMEOUT',error='Per-pass wall budget exhausted',seconds=time.monotonic()-started)
        except (EOFError,BrokenPipeError,OSError):
            self.stop()
            return dict(status='WORKER_EXIT',error='Isolated worker exited without a completed pass',seconds=time.monotonic()-started)

    def stop(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate();self.process.join(5)
            if self.process.is_alive():
                self.process.kill();self.process.join(5)
            if self.process.is_alive():raise RuntimeError('Previous worker did not stop; refuse a second GPU process')
            self.process.join();self.process.close();self.process=None
        if self.conn is not None:self.conn.close();self.conn=None


def completed_response(folder,root,response=None):
    """The child commits a response before sending it, closing the parent-death/acknowledgement gap."""
    folder,root=Path(folder),Path(root).resolve()
    path=folder/'completion.json'
    if response is not None:
        files={}
        def visit(value):
            if isinstance(value,dict):
                for v in value.values():visit(v)
            elif isinstance(value,(list,tuple)):
                for v in value:visit(v)
            elif isinstance(value,str):
                try:
                    p=Path(value)
                    if p.is_absolute() and p.is_relative_to(root) and p.is_file():files[str(p.relative_to(root))]=file_sha(p)
                except (ValueError,OSError):pass
        visit(response)
        atomic_json(path,dict(payload=response,payload_sha=digest(response),artifacts=files))
    if not path.exists():return None
    value=json.loads(path.read_text())
    if digest(value['payload'])!=value['payload_sha']:raise ValueError('Corrupt worker completion')
    for relative,sha in value['artifacts'].items():
        p=(root/relative).resolve()
        if not p.is_relative_to(root) or not p.is_file() or file_sha(p)!=sha:raise ValueError('Corrupt worker completion artifact')
    return value['payload']
