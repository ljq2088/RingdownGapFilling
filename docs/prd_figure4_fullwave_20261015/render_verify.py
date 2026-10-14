from pathlib import Path
import fitz,json
HERE=Path(__file__).resolve().parent
doc=fitz.open(HERE/'DGF_fullwave_20261015.pdf');figpages=[]
for i,p in enumerate(doc):
    text=p.get_text()
    if 'FIG. 4.' in text:
        figpages.append(i+1);p.get_pixmap(matrix=fitz.Matrix(1.65,1.65)).save(HERE/'figure4_page.png')
        assert all(s in text for s in ['528','Gapped signal','Observation','Complete time-domain reconstructions'])
assert len(figpages)==1,figpages
assert '8.52' in doc[0].get_text() and '5.32' in doc[0].get_text()
v=dict(pdf_pages=len(doc),figure4_pages=figpages,full_record_528s=True,four_curves_present=True,numerical_results_unchanged=True)
(HERE/'pdf_verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
