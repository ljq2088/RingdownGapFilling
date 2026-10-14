from pathlib import Path
import re,zipfile,json,hashlib,fitz
here=Path(__file__).resolve().parent
pdf=here/'overleaf_final.pdf'
if not pdf.exists():pdf=here/'local_manuscript.pdf'
d=fitz.open(pdf);text='\n'.join(p.get_text() for p in d)
assert all(len(p.get_text().strip())>250 for p in d),'Unexpected blank/near-empty page'
figures={};appendix_page=None;discussion_page=None
for i,p in enumerate(d):
 s=p.get_text()
 if 'DISCUSSION AND CONCLUSIONS' in s:discussion_page=i+1
 if 'Appendix A:' in s:appendix_page=i+1
 for n in re.findall(r'FIG\. (\d+)\.',s):
  assert not [x for x in p.get_links() if x.get('uri')],(i,n)
  figures[int(n)]={'page':i+1}
  if n in ['4','5','6','7','8']:p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(here/('figure'+n+'_page.png'))
assert len(figures)==8 and max(v['page'] for v in figures.values())<appendix_page,(figures,discussion_page,appendix_page)
assert 'Additional waveform diagnostics' not in text and '??' not in text
new=(here/'source/DGF.tex').read_text()
with zipfile.ZipFile(here/'overleaf_before.zip') as z:
 old=z.read('DGF.tex').decode('utf-8')
 tables=lambda x:re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',x,re.S)
 assert tables(old)==tables(new)
 assets={}
 for name in re.findall(r'\\includegraphics\[[^]]*\]\{([^}]+)\}',new):
  contents=(here/'source'/name).read_bytes();assert contents==z.read(name),name
  assets[name]=hashlib.sha256(contents).hexdigest()
report={'pdf':str(pdf),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'pages':len(d),'figures':figures,'discussion_page':discussion_page,'appendix_page':appendix_page,'all_figures_before_appendices':True,'result_tables_unchanged':True,'all_figure_files_unchanged':True,'figure_sha256':assets,'figure_number_mapping':{'old_7_SNR_gap':'new_4','old_4_waveform':'new_5','old_8_log_amplitude':'new_7','old_5_residual_PSD':'new_8','old_6_phase':'new_6'}}
if (here/'overleaf_after.zip').exists():
 with zipfile.ZipFile(here/'overleaf_after.zip') as z:
  assert z.read('DGF.tex')==(here/'source/DGF.tex').read_bytes()
  for name in assets:assert z.read(name)==(here/'source'/name).read_bytes(),name
 report['server_source_matches']=True
(here/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
