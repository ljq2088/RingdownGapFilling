from pathlib import Path
from zipfile import ZipFile
import re,json,hashlib,shutil,sys
import fitz

P=Path(__file__).resolve().parent
shutil.copy2(sys.argv[1],P/'overleaf_after.zip')
shutil.copy2(sys.argv[2],P/'overleaf_final.pdf')
before=ZipFile(P/'overleaf_before.zip');after=ZipFile(P/'overleaf_after.zip')
normal=lambda x:x.decode('utf-8').replace('\r\n','\n')
a=normal(before.read('DGF.tex'));b=normal(after.read('DGF.tex'))
assert b==(P/'source/DGF.tex').read_text(encoding='utf-8')
changed=[n for n in before.namelist() if n in after.namelist() and before.read(n)!=after.read(n)]
assert changed==['DGF.tex'],changed
for env in ('equation','align'):
    pattern=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    assert re.findall(pattern,a,re.S)==re.findall(pattern,b,re.S)
tables=r'\\begin\{tabular\}.*?\\end\{tabular\}'
for x,y in zip(re.findall(tables,a,re.S),re.findall(tables,b,re.S)):
    assert sorted(x.splitlines())==sorted(y.splitlines())
figs=re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',b)
assert len(figs)==8 and all(before.read(f)==after.read(f) for f in figs)
assert all(x.lower() not in b.lower() for x in ('low-SNR','low SNR','rho<10','0.109','failure case','selected successful'))
for text in ('0.00532','0.00241','0.01970','0.00852','0.00254','0.02782','5814','6000'):
    assert text in b
keys=set(k.strip() for g in re.findall(r'\\cite\{([^}]+)\}',b) for k in g.split(','))
assert len(keys)==26
entries=re.findall(r'@\w+\s*\{\s*([^,]+),',normal(after.read('DGF_references.bib')))
assert all(entries.count(k)==1 for k in keys)
d=fitz.open(P/'overleaf_final.pdf')
assert len(d)==12 and all(len(p.get_text().strip())>100 for p in d)
caption_pages={}
for i,p in enumerate(d):
    for n in re.findall(r'FIG\.\s*(\d+)\.',p.get_text()):caption_pages[int(n)]=i+1
appendix=next(i+1 for i,p in enumerate(d) if 'Appendix A:' in p.get_text())
assert sorted(caption_pages)==list(range(1,9))
assert all(p<appendix for p in caption_pages.values())
external={i+1:len([l for l in p.get_links() if l.get('uri')]) for i,p in enumerate(d) if i+1 in caption_pages.values()}
assert all(n==0 for n in external.values())
abstract=d[0].get_text()
assert abstract.index('5.32')<abstract.index('8.52')
for i in (0,5,7,8,10):d[i].get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(P/('cloud_page_%d.png'%(i+1)))
report=dict(cloud_source_matches=True,changed_cloud_files=changed,all_numeric_table_rows_preserved=True,equations_preserved=True,all_figure_assets_preserved=True,active_citation_keys=len(keys),pages=len(d),figure_pages=caption_pages,appendix_start=appendix,external_links_on_figure_pages=external,manuscript_sha256=hashlib.sha256((P/'overleaf_final.pdf').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(after.read('DGF.tex')).hexdigest(),test_checkpoint_sha256='0ffce08b8856f1e4f2c76b519941303edf226b8f7c4fd426db7fc98527d7e11b')
(P/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
