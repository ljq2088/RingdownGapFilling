from pathlib import Path
import shutil,os,subprocess,json
here=Path(__file__).resolve().parent
stage=Path(os.environ['TEMP'])/'ringdown_reviewer_revision_20261015'
stage.mkdir(exist_ok=True)
shutil.copytree(here/'source',stage/'source',dirs_exist_ok=True)
cmd=['python',str(Path.home()/'.codex/plugins/cache/openai-bundled/latex/0.2.8/scripts/compile_latex.py'),str(stage/'source/DGF.tex'),'--compiler','texlive','--output-directory',str(stage/'build'),'--json']
r=subprocess.run(cmd,capture_output=True)
(here/'compile_report.json').write_bytes(r.stdout)
(here/'compile_stderr.txt').write_bytes(r.stderr)
print(r.returncode,r.stderr.decode('utf-8','replace')[-1000:])
if r.returncode: print(r.stdout.decode('utf-8','replace')[-3000:])
pdf=stage/'build/DGF.pdf'
if pdf.exists():shutil.copy2(pdf,here/'local_manuscript.pdf');print(pdf)



