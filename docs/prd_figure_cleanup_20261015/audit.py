from pathlib import Path
import json,zipfile,shutil,hashlib
import fitz
HERE=Path(__file__).resolve().parent
src=HERE/'source';src.mkdir(exist_ok=True)
backup=Path.home()/'Downloads/RingdownGapFilling___PRD_revision_2026_10_09 (20).zip'
shutil.copy2(backup,HERE/'overleaf_before.zip')
with zipfile.ZipFile(backup) as z:z.extractall(src)
old=HERE.parent/'waveform_views_20261015/source/DGF.tex'
assert old.read_bytes()==(src/'DGF.tex').read_bytes(), 'Cloud changed; inspect before editing'
pdf=fitz.open(HERE.parent/'manuscript_fig_review_20261015/latest_overleaf.pdf')
result={}
for name in ['latest_diagnostic_examples.pdf','source_polarizations_20261015.pdf']:
 d=fitz.open(src/name);result[name]={'links':d[0].get_links(),'sha256':hashlib.sha256((src/name).read_bytes()).hexdigest()}
result['full_pdf_external_links']={str(i+1):[{'rect':list(l['from']),'uri':l.get('uri')} for l in p.get_links() if l.get('uri') and l['from'].y0<500] for i,p in enumerate(pdf) if i>=10}
(HERE/'provenance_links.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2)[:5000])
