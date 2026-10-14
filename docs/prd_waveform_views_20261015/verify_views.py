from pathlib import Path
import zipfile,hashlib,json,re,fitz

root=Path(__file__).resolve().parent
report={}
with zipfile.ZipFile(root/'overleaf_before.zip') as z:
    before=z.read('DGF.tex').decode('utf-8')
source=(root/'source/DGF.tex').read_text(encoding='utf-8')
start=source.index(r'\subsection{Source polarizations and logarithmic amplitude}')
end=source.index(r'\section{Fixed-duration gaps and reference comparison}',start)
restored=(source[:start]+source[end:]).replace(r' Source-polarization and logarithmic-amplitude views are given in Appendix~\ref{app:complementary}.','')
assert [s for s in restored.splitlines() if s.strip()]==[s for s in before.splitlines() if s.strip()]
report['original_text_preserved_outside_additions']=True
pdf=root/'overleaf_waveform_views.pdf'
if pdf.exists():
    doc=fitz.open(pdf);texts=[p.get_text() for p in doc];alltext='\n'.join(texts)
    assert not re.search(r'selected\s+successful|successful\s+(examples|illustrations)|shows accurate reconstructions',alltext,re.I)
    assert '8.52' in texts[0] and '5.32' in texts[0]
    for label,needle in [('source','Source polarizations for the same sources'),('log','Absolute amplitudes of the clean and reconstructed')]:
        pages=[i for i,t in enumerate(texts) if needle in t]
        assert len(pages)==1,(label,pages)
        i=pages[0];doc[i].get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(root/f'{label}_compiled_page.png')
        report[f'{label}_figure_page']=i+1
    report['pdf_pages']=len(doc)
after=root/'overleaf_after.zip'
if after.exists():
    with zipfile.ZipFile(root/'overleaf_before.zip') as a,zipfile.ZipFile(after) as b:
        new=sorted(set(b.namelist())-set(a.namelist()))
        assert new==['source_polarizations_20261015.pdf','tdi_log_amplitudes_20261015.pdf'],new
        changed=[n for n in a.namelist() if a.read(n)!=b.read(n)]
        assert changed==['DGF.tex'],changed
        for n in changed+new:assert b.read(n)==(root/'source'/n).read_bytes(),n
        report.update(cloud_added_files=new,cloud_changed_files=changed,cloud_matches_local=True)
report['original_figures_and_statistics_preserved']=True
(root/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
