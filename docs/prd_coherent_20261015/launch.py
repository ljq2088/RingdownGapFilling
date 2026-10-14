from pathlib import Path
import subprocess,os,json
root=Path('/home/ljq/code/RingdownGapFilling_PRD');out=root/'FillingGap/results_prd_coherent_20261015'
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='1')
with open(out/'pipeline.log','ab',buffering=0) as log:
    p=subprocess.Popen(['/home/ljq/miniconda3/envs/gw_ml_env/bin/python','-u','-m','prd_coherent.run'],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'launch.json').write_text(json.dumps(dict(pid=p.pid,command='python -u -m prd_coherent.run',log=str(out/'pipeline.log')),indent=2));print(p.pid)
