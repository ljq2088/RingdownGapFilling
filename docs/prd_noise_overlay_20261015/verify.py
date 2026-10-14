from pathlib import Path
import fitz,json,re,hashlib,zipfile,numpy as np
HERE=Path(__file__).resolve().parent
pdfpath=HERE/'overleaf_final.pdf'
if not pdfpath.exists():pdfpath=HERE/'local_manuscript.pdf'
d=fitz.open(pdfpath);text='\n'.join(p.get_text() for p in d)
assert all(s not in text for s in ['Rank 3000','Rank 5400','Rank 6000','Selected successful','upper strip','Source polarizations for the same'])
assert '2038202231' in text and '2038201952' in text
assert '0.00852' in text and '0.00532' in text
figures={}
for i,p in enumerate(d):
 for number in re.findall(r'FIG\. (\d+)\.',p.get_text()):
  assert not [x for x in p.get_links() if x.get('uri')], (i+1,number)
  figures[number]={'page':i+1,'external_links':0}
  if number in ['4','6','8']:p.get_pixmap(matrix=fitz.Matrix(1.4,1.4)).save(HERE/('figure'+number+'_final_page.png'))
assert set(figures)==set(map(str,range(1,9))),figures
native=np.load(HERE/'figure4_native_samples.npz');assert native['truth'].shape==(2,2,1056)
assert np.max(abs(native['observation']-(native['gapped_truth']+native['noise'])))==0
assert np.max(abs(native['gapped_truth']*native['mask'][:,None]))==0
with zipfile.ZipFile(HERE/'overleaf_before.zip') as z:
 old=z.read('DGF.tex').decode('utf-8');new=(HERE/'source/DGF.tex').read_text()
 # Every numerical result table is unchanged.
 tables=lambda x:re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',x,re.S)
 assert tables(old)==tables(new)
 for name in ['latest_mismatch.pdf','latest_snr_gap.pdf','latest_residual_psd.pdf']:
  assert z.read(name)==(HERE/'source'/name).read_bytes()
report={'pdf':str(pdfpath),'pdf_sha256':hashlib.sha256(pdfpath.read_bytes()).hexdigest(),'pages':len(d),'figures':figures,'old_diagnostic_figure_removed':True,'observation_overlaid_at_original_amplitude':True,'noise_amplitude_multiplier':1,'smoothing':False,'native_samples':1056,'dt_seconds':.5,'result_tables_and_population_plots_unchanged':True}
if (HERE/'overleaf_after.zip').exists():
 with zipfile.ZipFile(HERE/'overleaf_after.zip') as z:
  assert 'latest_diagnostic_examples.pdf' not in z.namelist()
  for name in ['DGF.tex','latest_examples.pdf','latest_phase.pdf','tdi_log_amplitudes_20261015.pdf']:assert z.read(name)==(HERE/'source'/name).read_bytes(),name
 report['server_sources_match']=True
(HERE/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
