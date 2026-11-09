"""Small GPU acceptance check using released checkpoints; no training or evaluation."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.model_score import score_model
from scripts.skill_package import read_package
from scan import scan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output file')
    import torch
    import transformers
    import huggingface_hub
    import importlib.metadata
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    report={'python':platform.python_version(),'torch':torch.__version__,
            'transformers':transformers.__version__,'huggingface_hub':huggingface_hub.__version__,
            'device':torch.cuda.get_device_name(0),'vram_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
            'scope':'Inference acceptance only; project-authored fixtures, no training or benchmark evaluation.',
            'files_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scan.py','scripts/model_score.py','scripts/package_inputs.py','scripts/skill_package.py','configs/community-models.json']},'cases':[]}
    try:report['flash_linear_attention']=importlib.metadata.version('flash-linear-attention')
    except importlib.metadata.PackageNotFoundError:report['flash_linear_attention']=None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    for mode in ('markdown','package'):
        path=ROOT/'examples/local-inventory'
        if mode=='markdown':path=path/'SKILL.md'
        for length in ('short','context_limit'):
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            if length=='short':
                result=scan(path)['model']
            else:
                row,_=read_package(path)
                filler='Synthetic documentation for an offline inventory example. Count file types and print a summary.\n'*4000
                if mode=='markdown':row['skill_text']+= '\n'+filler
                else:row['files'].append({'path':'references/long-fixture.txt','text':filler})
                result=score_model(row,mode)
            torch.cuda.synchronize()
            if result['status']!='complete' or not 0<=result['score']<=1:raise RuntimeError(str(result))
            if length=='context_limit':
                assert result['coverage']['truncated']
                assert result['coverage']['used_tokens']==(8192 if mode=='markdown' else 32768)
            case={'mode':mode,'length':length,'seconds_including_model_load':time.perf_counter()-started,
                  'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                  'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'result':result}
            report['cases'].append(case);save()
            print(json.dumps({k:v for k,v in case.items() if k!='result'}),flush=True)
    report['passed']=True;save()


if __name__=='__main__':main()
