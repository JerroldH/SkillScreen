"""Recover pinned community inputs into inert JSONL. Never extract or run packages."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import re
import stat
import tarfile
from urllib.parse import quote, urlsplit
import urllib.request
import shutil
import zipfile

try:
    from .community_data import DEFAULT_DATA, read_index, sha256
    from .import_malskillbench import normalized_hash
    from .package_inputs import safe_relative_path
except ImportError:
    from community_data import DEFAULT_DATA, read_index, sha256
    from import_malskillbench import normalized_hash
    from package_inputs import safe_relative_path


def cached_file(url, digest, cache, offline=False):
    """Cache raw bytes by SHA256, without preserving executable source filenames."""
    parsed = urlsplit(url)
    if not re.fullmatch('[0-9a-f]{64}',digest):
        raise ValueError('Expected a SHA256 cache key')
    if parsed.scheme != 'https' or parsed.username or parsed.password or parsed.hostname not in {'huggingface.co', 'raw.githubusercontent.com'}:
        raise ValueError('Expected a pinned public HF or GitHub download')
    path = cache/(digest+'.blob')
    if not path.exists():
        if offline:
            raise ValueError('Missing offline cache: '+digest)
        cache.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.part')
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'SkillScreen-data/0.1'}),timeout=60) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source,target)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest:
                raise ValueError('Downloaded source hash mismatch')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('Cached source hash mismatch: '+digest)
    return path


def restore(rows, sources, mode, cache, offline=False):
    """Validate complete inventories and preserve the canonical Markdown exactly."""
    if mode not in ('markdown','package'):
        raise ValueError('Choose markdown or package')
    cache = Path(cache)
    with ExitStack() as stack:
        archives, primary = {}, None
        for row in rows:
            if mode=='package' and row['coverage']!='package_complete':
                raise ValueError('Select the explicit package_complete cohort first')
            recovery = row['recovery']['markdown']
            if 'asset' in recovery:
                if primary is None:
                    import pyarrow.parquet as pq
                    asset=sources[recovery['asset']]
                    path=cached_file(asset['url'],asset['sha256'],cache,offline)
                    primary={r['benchmark_id']:r['skill_text'] for r in pq.read_table(path,columns=['benchmark_id','skill_text']).to_pylist()}
                markdown=primary[recovery['benchmark_id']]
            else:
                markdown=cached_file(recovery['url'],recovery['sha256'],cache,offline).read_bytes().decode('utf-8')
            md=next(f for f in row['files'] if f['path']=='SKILL.md')
            if not isinstance(markdown,str) or sha256(markdown)!=md['sha256']:
                raise ValueError('Canonical Markdown mismatch: '+row['id'])
            companions=[]
            if mode=='package':
                package=row['recovery']['package']
                if 'asset' in package:
                    key=package['asset']
                    if key not in archives:
                        asset=sources[key]
                        path=cached_file(asset['url'],asset['sha256'],cache,offline)
                        if asset['format']=='zip':
                            archive=stack.enter_context(zipfile.ZipFile(path))
                            members={}
                            for item in archive.infolist():
                                if item.is_dir():continue
                                safe_relative_path(item.filename)
                                if item.filename in members or stat.S_ISLNK(item.external_attr>>16):
                                    raise ValueError('Linked or duplicate archive member')
                                members[item.filename]=item
                        else:
                            archive=stack.enter_context(tarfile.open(path,'r:gz'))
                            members={}
                            for item in archive:
                                if item.isdir():continue
                                safe_relative_path(item.name)
                                if not item.isfile() or item.name in members:
                                    raise ValueError('Linked or duplicate archive member')
                                members[item.name]=archive.extractfile(item).read()
                        archives[key]=(archive,members)
                    archive,members=archives[key]
                    actual={name for name in members if name.startswith(package['prefix'])}
                    expected={f['source_path'] for f in row['files']}
                    if actual!=expected:
                        raise ValueError('Complete package inventory mismatch: '+row['id'])
                for item in row['files']:
                    safe_relative_path(item['source_path'])
                    if 'asset' in package:
                        member=members[item['source_path']]
                        raw=archive.read(member) if isinstance(archive,zipfile.ZipFile) else member
                    else:
                        url=package['base_url']+quote(item['source_path'],safe='/')
                        raw=cached_file(url,item['source_sha256'],cache,offline).read_bytes()
                    if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['source_sha256']:
                        raise ValueError('Original package file mismatch: '+row['id']+'/'+item['path'])
                    if item['path']=='SKILL.md':
                        if normalized_hash(raw.decode('utf-8-sig'))!=normalized_hash(markdown):
                            raise ValueError('Package/Markdown identity mismatch')
                    elif item['input_eligible']:
                        text=raw.decode('utf-8-sig')
                        if sha256(text)!=item['sha256']:
                            raise ValueError('Companion input hash mismatch')
                        companions.append({'path':item['path'],'text':text})
            yield {'id':row['id'],'label':int(row['label']=='malicious'),'skill_text':markdown,'files':companions}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--mode',choices=('markdown','package'),required=True)
    parser.add_argument('--cache',type=Path,default=Path('outputs/community-cache'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--id',action='append',help='Recover specified record IDs only')
    parser.add_argument('--offline',action='store_true',help='Require previously verified download cache')
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.data.resolve()) or args.cache.resolve().is_relative_to(args.data.resolve()):
        parser.error('Keep new output and raw cache outside the public data directory')
    index=read_index(args.data)
    rows=list(index.values()) if not args.id else [index[key] for key in args.id]
    unavailable=[r for r in rows if args.mode=='package' and r['coverage']!='package_complete']
    if args.id and unavailable:
        parser.error('A requested ID has no complete package')
    rows=[r for r in rows if r['coverage']=='package_complete'] if args.mode=='package' else rows
    sources=json.loads((args.data/'sources.json').read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+'.part')
    handle=temporary.open('x',encoding='utf-8',newline='\n')
    try:
        with handle:
            for count,row in enumerate(restore(rows,sources,args.mode,args.cache,args.offline),1):
                handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                if count%1000==0:print(f'{count}/{len(rows)} records verified',flush=True)
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({'restored':len(rows),'mode':args.mode,'unavailable_package_coverage':dict(Counter(r['coverage'] for r in unavailable)),
                      'output':str(args.output),'scope':'Inputs only. No training, package execution, or upload.'},indent=2))


if __name__=='__main__':
    main()
