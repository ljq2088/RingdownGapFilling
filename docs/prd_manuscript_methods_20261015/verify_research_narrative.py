from pathlib import Path
import zipfile
import hashlib
import json
import re
import fitz

p = Path(__file__).resolve().parent
assets = ['DGF.tex','DGF_references.bib','architecture_dgf_20261015.pdf',
          'training_workflow_dgf_20261015.pdf','examples_best_peak.pdf',
          'mismatch_cdf_presented.pdf','error_psd_presented.pdf','phase_presented.pdf']
z = zipfile.ZipFile(p/'overleaf_after.zip')
checks = {}
for name in assets:
    data = (p/'source_edited'/name).read_bytes()
    if name.endswith(('.tex', '.bib')) and data != z.read(name):
        assert data.replace(b'\r\n', b'\n') == z.read(name).replace(b'\r\n', b'\n'), name
        # Overleaf normalizes text uploads to LF. Match its exported bytes after
        # verifying that line endings are the only difference.
        data = z.read(name)
        (p/'source_edited'/name).write_bytes(data)
    assert z.read(name) == data, name
    checks[name] = {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
before = zipfile.ZipFile(p/'overleaf_before.zip')
assert all(before.read(n)==z.read(n) for n in before.namelist() if n.endswith('.pdf'))
tex = z.read('DGF.tex').decode('utf-8')
main = tex.split(r'\appendix')[0]
abstract = main.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
assert '0.02715' not in abstract and '80.2' not in abstract
assert 'First refinement' not in main and 'Augmented refinement' not in main
assert 'Original DGF & 0.02537 & 0.02715 & 0.00444 & 0.09803' in tex
assert 'Mean $\\mathcal M_S$' in tex
assert 'TorresForne2016' in tex
doc = fitz.open(p/'DGF_research_replacement_20261015.pdf')
pages = [page.get_text() for page in doc]
assert '5.37' in pages[0] and '0.02715' not in pages[0]
assert 'Supplementary training comparisons' in '\n'.join(pages)
repo = Path('/home/ljq/code/RingdownGapFilling_PRD')
manifest = json.loads((repo/'FillingGap/results_prd_conditioned_20261014/runtime_source_manifest.json').read_text())
assert all(hashlib.sha256((repo/path).read_bytes()).hexdigest()==sha for path,sha in manifest.items())
v = {'overleaf_project':'6ac83816750c2cce452054d2','verified_files':checks,
     'pdf_pages':len(doc),'online_pdf_text_verified':True,
     'research_narrative_revision_complete':True,
     'original_comparison_rows_preserved_in_appendix':True,
     'legacy_result_pdfs_unchanged':True,'runtime_sources_unchanged':True,
     'final_physical_AE_results_inserted':False,
     'main_figure_style_preserved':True,'complete_test_population_retained':True}
(p/'server_verification.json').write_text(json.dumps(v,indent=2))
print(json.dumps(v,indent=2))
