from pathlib import Path
import zipfile,fitz,json,hashlib,re
P=Path(__file__).resolve().parent
A=zipfile.ZipFile(P/'overleaf_before.zip');B=zipfile.ZipFile(P/'overleaf_after.zip')
changed=[n for n in B.namelist() if n not in A.namelist() or A.read(n)!=B.read(n)]
assert set(changed)=={'DGF.tex','reproducibility_manifest.json'},changed
for n in changed:assert B.read(n)==(P/'source'/n).read_bytes(),n
before=A.read('DGF.tex').decode('utf8');after=B.read('DGF.tex').decode('utf8')
tables=lambda x:re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}',x,re.S)
assert tables(before)==tables(after)
for label in ['eq:observation','eq:whitening','eq:basis_adapter','eq:refinement','eq:loss']:
    blocks=lambda x:[m.group() for m in re.finditer(r'\\begin\{(equation|align)\}.*?\\end\{\1\}',x,re.S) if label in m.group()]
    assert blocks(before)==blocks(after),label
m=json.loads(B.read('reproducibility_manifest.json'))
assert not (set(m)&{'second_review_audit','initialization_provenance_status'})
for word in ['next validation targets','Application to missing TDI samples requires','Generation requires','aborts if','maximum discrete gains','Selected successful','initialize reproduction']:
    assert word not in after,word
d=fitz.open(P/'overleaf_final.pdf');assert len(d)==13
txt='\n'.join(pg.get_text() for pg in d)
for val in ['0.00532','0.00852','0.04131','4.92']:assert val in txt,val
figure_pages=[i+1 for i,pg in enumerate(d) if re.search(r'FIG\. [1-8]\.',pg.get_text())]
assert max(figure_pages)<11
external=[(i,l.get('uri')) for i in figure_pages for l in d[i-1].get_links() if l.get('uri')]
assert not external,external
for i in [0,2,3,5,7,8,9,10,11,12]:
    d[i].get_pixmap(matrix=fitz.Matrix(1.1,1.1)).save(P/f'cloud_qa_{i+1}.png')
report=json.loads((P/'compile_report.json').read_text(encoding='utf8'))
issues=re.findall(r'.*(?:Overfull|undefined|LaTeX Error).*','\n'.join(x.get('log','') for x in report['attempts']))
assert not issues,issues
record=dict(project='6ac83816750c2cce452054d2',pages=13,changed_files=changed,all_changed_files_match=True,numeric_tables_unchanged=True,core_equations_unchanged=True,all_eight_figure_assets_unchanged=True,review_specific_sections_removed=True,figure_pages=figure_pages,external_links_on_figure_pages=0,pdf_sha256=hashlib.sha256((P/'overleaf_final.pdf').read_bytes()).hexdigest(),files_sha256={n:hashlib.sha256(B.read(n)).hexdigest() for n in changed})
(P/'verification.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps(record,indent=2))
