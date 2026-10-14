from pathlib import Path
import fitz,json,re,hashlib,zipfile,numpy as np
HERE=Path(__file__).resolve().parent
pdfpath=HERE/'overleaf_final.pdf'
if not pdfpath.exists():pdfpath=HERE/'local_manuscript.pdf'
d=fitz.open(pdfpath);text='\n'.join(p.get_text() for p in d)
assert 'Source polarizations for the same' not in text
assert 'Selected successful' not in text
assert '2038205946' in text and '2038201952' in text and '2038202314' in text
assert '0.00852' in text and '0.00532' in text
figures={}
for i,p in enumerate(d):
 matches=re.findall(r'FIG\. (\d+)\.',p.get_text())
 if matches:
  rectangles=[list(x['from']) for x in p.get_links() if x.get('uri') and x['from'].height>30]
  assert not rectangles, (i,rectangles)
  for number in matches:
   figures[number]={'page':i+1,'external_links':len([x for x in p.get_links() if x.get('uri')])}
   assert figures[number]['external_links']==0, (number,figures[number])
   if number in ['4','6','8','9']:p.get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(HERE/('figure'+number+'_final_page.png'))
assert set(figures)==set(map(str,range(1,10))),figures
for n in ['latest_examples.pdf','latest_phase.pdf','latest_diagnostic_examples.pdf','tdi_log_amplitudes_20261015.pdf']:
 assert len(fitz.open(HERE/'source'/n)[0].get_links())==0
native=np.load(HERE/'figure4_native_samples.npz')
assert native['truth'].shape==(3,2,1056)
assert np.max(abs(native['observation']-(native['gapped_truth']+native['noise'])))==0
assert np.max(abs(native['gapped_truth']*native['mask'][:,None]))==0
report={'pdf':str(pdfpath),'pdf_sha256':hashlib.sha256(pdfpath.read_bytes()).hexdigest(),'pages':len(d),'figures':figures,'source_polarization_figure_removed':True,'figure_external_links':0,'native_samples':1056,'dt_seconds':.5,'statistics_unchanged':True}
if (HERE/'overleaf_after.zip').exists():
 with zipfile.ZipFile(HERE/'overleaf_after.zip') as z:
  assert 'source_polarizations_20261015.pdf' not in z.namelist()
  for n in ['DGF.tex','latest_examples.pdf','latest_phase.pdf','latest_diagnostic_examples.pdf','tdi_log_amplitudes_20261015.pdf']:
   assert z.read(n)==(HERE/'source'/n).read_bytes(),n
 report['server_sources_match']=True
(HERE/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
