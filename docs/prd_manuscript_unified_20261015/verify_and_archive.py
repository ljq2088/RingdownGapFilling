from pathlib import Path
import json,re,zipfile,hashlib,shutil,sys
import fitz

HERE=Path(__file__).resolve().parent
ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD')
sys.path.insert(0,str(ROOT))
from prd_conditioned.common import check
from prd_conditioned.settings import OUT as RUN, CAL
from prd_physical.common import sha

check()
report=json.loads((HERE/'latest_test_summary.json').read_text())
assert sha(RUN/'finetune/best.pth')==report['checkpoint_sha256']
assert all(sha(RUN/name)==value for name,value in report['provenance'].items())
z=zipfile.ZipFile(HERE/'overleaf_after.zip')
tex=z.read('DGF.tex').decode('utf-8')
files=['DGF.tex','DGF_references.bib']+re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',tex)
assert len(files)==10 and len(set(files))==10
verified={}
for name in files:
    data=(HERE/'source_edited'/name).read_bytes()
    if name.endswith(('.tex','.bib')) and data!=z.read(name):
        assert data.replace(b'\r\n',b'\n')==z.read(name).replace(b'\r\n',b'\n')
        data=z.read(name);(HERE/'source_edited'/name).write_bytes(data)
    assert data==z.read(name),name
    verified[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
before=zipfile.ZipFile(HERE/'overleaf_before.zip')
assert before.read('architecture_dgf_20261015.pdf')==z.read('architecture_dgf_20261015.pdf')
assert before.read('training_workflow_dgf_20261015.pdf')==z.read('training_workflow_dgf_20261015.pdf')
assert not any(word in tex for word in ['four-component','noise family','0.00537','0.02715','2026240','First refinement','Augmented refinement'])
doc=fitz.open(HERE/'DGF_unified_latest_20261015.pdf')
assert '8.52' in doc[0].get_text() and '5.32' in doc[0].get_text()
assert '6000' in ''.join(page.get_text() for page in doc).replace(',','')
v=dict(overleaf_project='6ac83816750c2cce452054d2',verified_files=verified,pdf_pages=len(doc),
       checkpoint_sha256=report['checkpoint_sha256'],test_sources=6000,seeds=report['seeds'],
       primary_mean=report['primary']['psd']['mean'],snr_ge10_mean=report['snr_ge10']['psd']['mean'],
       same_latest_population_for_all_reconstruction_results=True,
       calibration_psd_sha256=sha(CAL),training_sources_and_fixed_assets_unchanged=True,
       model_diagram_and_training_flow_unchanged=True,old_results_absent_from_manuscript=True,
       selected_case_rule='Three lowest joint noise-weighted mismatches within peak-inside-gap subgroup',
       selected_source_count=report['peak_in_gap']['n'],time_phase_alignment=False)
(HERE/'server_verification.json').write_text(json.dumps(v,indent=2))
dest=ROOT/'docs/prd_manuscript_unified_20261015'; dest.mkdir(exist_ok=True)
for name in ['evaluate_and_plot_latest.py','write_unified_manuscript.py','resume_final_evaluation.py',
             'render_unified.py','verify_and_archive.py','latest_test_summary.json',
             'unified_manuscript_checks.json','server_verification.json','UNIFICATION_REPORT.txt']:
    shutil.copy2(HERE/name,dest/name)
src=dest/'source_edited';src.mkdir(exist_ok=True)
for name in files:shutil.copy2(HERE/'source_edited'/name,src/name)
for name in files:
    if name.startswith('latest_'):
        svg=Path(name).with_suffix('.svg');shutil.copy2(HERE/'source_edited'/svg,src/svg)
# A compact clean source bundle contains only the files used by this manuscript.
with zipfile.ZipFile(HERE/'DGF_unified_source_20261015.zip','w',zipfile.ZIP_DEFLATED) as clean:
    for name in files:clean.write(HERE/'source_edited'/name,name)
print(json.dumps({k:v[k] for k in ['pdf_pages','test_sources','primary_mean','snr_ge10_mean','same_latest_population_for_all_reconstruction_results','training_sources_and_fixed_assets_unchanged']},indent=2))
