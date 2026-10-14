from pathlib import Path
from zipfile import ZipFile
import re,json,hashlib,shutil
import fitz
P=Path(__file__).resolve().parent
shutil.copy2(Path.home()/'Downloads/RingdownGapFilling___PRD_revision_2026_10_09 (28).zip',P/'overleaf_after.zip')
shutil.copy2(Path.home()/'Downloads/RingdownGapFilling___PRD_revision_2026_10_09 (27).zip',P/'overleaf_preupload.zip')
before=ZipFile(P/'overleaf_before.zip'); after=ZipFile(P/'overleaf_after.zip')
tex=after.read('DGF.tex').decode('utf-8')
original=before.read('DGF.tex').decode('utf-8')
def norm(s):return s.replace('\r\n','\n')
cloud={n:norm(after.read(n).decode('utf-8'))==norm((P/'source'/n).read_text(encoding='utf-8')) for n in ('DGF.tex','DGF_references.bib')}
assert all(cloud.values()),cloud
equations={}
for env in ('equation','align','tabular'):
    pattern=r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}'
    a=re.findall(pattern,norm(original),re.S);b=re.findall(pattern,norm(tex),re.S)
    equations[env]=dict(count=len(a),unchanged=a==b)
    assert a==b
figs=re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',tex)
figures={f:hashlib.sha256(after.read(f)).hexdigest() for f in figs}
assert all(before.read(f)==after.read(f) for f in figs)
assert len(figs)==8
keys=sorted(set(k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',tex) for k in group.split(',')))
bib=after.read('DGF_references.bib').decode('utf-8')
entries=re.findall(r'@\w+\s*\{\s*([^,]+),',bib)
missing=[k for k in keys if k not in entries];duplicates=[k for k in keys if entries.count(k)!=1]
assert not missing and not duplicates
assert 'pages={044035}' in bib and 'Yoder, Nate' in bib
assert all(w.lower() not in tex.lower() for w in ['Selected successful reconstruction','four-component','0.00537','evaluated once','white-space RMS'])
pdf=fitz.open(P/'overleaf_final.pdf')
caption_pages={}
for i,page in enumerate(pdf):
    for number in re.findall(r'FIG\.\s*(\d+)\.',page.get_text()):caption_pages[int(number)]=i+1
app_page=next(i+1 for i,p in enumerate(pdf) if 'Appendix A:' in p.get_text())
assert all(page<app_page for page in caption_pages.values())
assert sorted(caption_pages)==list(range(1,9)),caption_pages
assert len(pdf)==12 and all(len(page.get_text().strip())>100 for page in pdf)
external_on_figures={i+1:len([l for l in page.get_links() if l.get('uri')]) for i,page in enumerate(pdf) if i+1 in caption_pages.values()}
assert all(n==0 for n in external_on_figures.values())
review=fitz.open(P/'manuscript_review.pdf')
assert len(review)==4
# Render a complete contact sheet for final layout inspection.
from PIL import Image,ImageOps,ImageDraw
thumbs=[]
for i,page in enumerate(pdf):
    pix=page.get_pixmap(matrix=fitz.Matrix(.55,.55))
    im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
    thumb=Image.new('RGB',(345,490),'#dde4e9');thumb.paste(im,((345-im.width)//2,10))
    ImageDraw.Draw(thumb).text((10,475),str(i+1),fill='black');thumbs.append(thumb)
sheet=Image.new('RGB',(345*4,490*3),'white')
for i,im in enumerate(thumbs):sheet.paste(im,((i%4)*345,(i//4)*490))
sheet.save(P/'all_pages.png')
v=dict(cloud_source_matches=cloud,figure_files_sha256=figures,equations_and_tables=equations,cited_keys_count=len(keys),missing_citations=missing,duplicate_active_keys=duplicates,pages=len(pdf),figure_pages=caption_pages,appendix_starts=app_page,external_links_on_figure_pages=external_on_figures,manuscript_sha256=hashlib.sha256((P/'overleaf_final.pdf').read_bytes()).hexdigest(),review_sha256=hashlib.sha256((P/'manuscript_review.pdf').read_bytes()).hexdigest(),review_pages=len(review),cloud_build='1a13cb859b6-f1c4c9fac6906ed8',test_checkpoint_sha256='0ffce08b8856f1e4f2c76b519941303edf226b8f7c4fd426db7fc98527d7e11b')
(P/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
print(json.dumps(v,indent=2))
