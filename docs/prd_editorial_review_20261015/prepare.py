from pathlib import Path
import zipfile,shutil,re,json,hashlib
here=Path(__file__).resolve().parent
backup=Path.home()/'Downloads/RingdownGapFilling___PRD_revision_2026_10_09 (26).zip'
shutil.copy2(backup,here/'overleaf_before.zip')
with zipfile.ZipFile(backup) as z:z.extractall(here/'source');tex=z.read('DGF.tex').decode('utf-8')
prior=(here.parent/'main_text_figures_20261015/source/DGF.tex').read_text(encoding='utf-8')
(here/'initial_source_status.json').write_text(json.dumps({'matches_preceding_turn':tex==prior,'sha256':hashlib.sha256(tex.encode()).hexdigest()},indent=2))
bib=(here/'source/DGF_references.bib').read_text(encoding='utf-8');entries=[]
for m in re.finditer(r'@\w+\s*\{\s*([^,]+),',bib):
 depth=1;j=m.end()
 while j<len(bib) and depth:
  if bib[j]=='{' and (j==0 or bib[j-1]!='\\'):depth+=1
  elif bib[j]=='}' and (j==0 or bib[j-1]!='\\'):depth-=1
  j+=1
 entries.append((m.group(1).strip(),bib[m.start():j]))
keys=set()
for m in re.finditer(r'\\cite\w*\{([^}]+)\}',tex):keys.update(x.strip() for x in m.group(1).split(','))
active=[(k,e) for k,e in entries if k in keys]
(here/'cited_bibliography.bib').write_text('\n\n'.join(e for k,e in active),encoding='utf-8')
(here/'citation_inventory.json').write_text(json.dumps({'active_keys':sorted(keys),'matched_keys':sorted(k for k,e in active),'missing':sorted(keys-{k for k,e in active}),'duplicate_active_keys':[k for k in keys if sum(kk==k for kk,e in entries)>1]},indent=2))
print('Fresh cloud source matches preceding turn:',tex==prior)
print('Citations:',len(keys),'matched:',len(active))
print((here/'citation_inventory.json').read_text())
