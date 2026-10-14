from pathlib import Path
import zipfile,json,hashlib,shutil,sys
import re
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_coherent.common import check
check()
before=zipfile.ZipFile(HERE/'overleaf_before.zip');after=zipfile.ZipFile(HERE/'overleaf_after.zip')
assert set(before.namelist())==set(after.namelist())
changed=[n for n in before.namelist() if before.read(n)!=after.read(n)]
assert sorted(changed)==['DGF.tex','latest_examples.pdf','latest_phase.pdf'],changed
for n in changed:
    local=(HERE/'source_edited'/n).read_bytes();remote=after.read(n)
    assert local==remote or (n.endswith('.tex') and local.replace(b'\r\n',b'\n')==remote.replace(b'\r\n',b'\n'))
tex=after.read('DGF.tex').decode();assert r'\ref{fig:examples}(a)' in tex
report=dict(overleaf_project='6ac83816750c2cce452054d2',changed_cloud_files=changed,all_other_cloud_files_unchanged=True,local_remote_contents_match=True,checkpoint_and_training_assets_unchanged=True,source_sha={n:hashlib.sha256(after.read(n)).hexdigest() for n in changed})
(HERE/'cloud_verification.json').write_text(json.dumps(report,indent=2))
dest=ROOT/'docs/prd_figure4_oscillatory_20261015';dest.mkdir(exist_ok=True)
for name in ['figure4_audit.json','pdf_verification.json','cloud_verification.json','render_verify.py','verify_archive.py']:shutil.copy2(HERE/name,dest/name)
shutil.copy2(HERE.parent/'optimization_round_20261015/figure4_best_oscillatory.py',dest/'figure4_best_oscillatory.py')
for n in changed:shutil.copy2(HERE/'source_edited'/n,dest/n)
shutil.copy2(HERE/'source_edited/latest_examples.svg',dest/'latest_examples.svg')
shutil.copy2(HERE/'source_edited/latest_phase.svg',dest/'latest_phase.svg')
with zipfile.ZipFile(HERE/'DGF_oscillatory_source_20261015.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['DGF.tex','DGF_references.bib']+re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',tex):z.writestr(name,after.read(name))
print(json.dumps(report,indent=2))
