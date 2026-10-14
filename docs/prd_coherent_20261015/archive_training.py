from pathlib import Path
import json,sys,shutil,hashlib,time
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/ljq/code/RingdownGapFilling_PRD');sys.path.insert(0,str(ROOT))
from prd_coherent.settings import OUT
from prd_coherent.common import check
check();dest=ROOT/'docs/prd_coherent_20261015';dest.mkdir(exist_ok=True)
for name in ['analyze.py','analysis.json','RESEARCH_AND_TRAINING.md','bootstrap.py','launch.py','check_preflight.py','report_status.py','archive_training.py','audit_augmentation.py','diagnose_waveform_shape.py','waveform_shape_diagnosis.json','trace_source_tdi.py','test_onset_sensitivity.py','WAVEFORM_PHYSICS_AUDIT.txt','find_oscillatory_examples.py','oscillatory_candidates.json','selected_oscillatory_examples.json','figure4_best_oscillatory.py']:
    shutil.copy2(HERE/name,dest/name)
for name in ['smoke.json','additional_preflight.json','protocol.json','runtime_source_manifest.json','asset_manifest.json','training_snr_summary.json','mask_gram_complete.json','pilot_selection.json','augmentation_audit.json']:
    if (OUT/name).exists():shutil.copy2(OUT/name,dest/name)
revision=dict(time=time.time(),changes=['Stage-completion logging uses stage_name without shadowing the status stage argument.','Resume handles an already completed stopping boundary.','Resume preserves the scheduled noise augmentation generation.','Noise-feature preparation preserves the training RNG.','Cache budget adds 0.5 GiB desktop headroom.'],gradient_objective_unchanged=True,completed_epochs_and_checkpoints_retained=True,final_test_read=False)
(OUT/'preflight_revision/revision.json').write_text(json.dumps(revision,indent=2));shutil.copy2(OUT/'preflight_revision/revision.json',dest/'preflight_revision.json')
for folder in ('waveform_trace','waveform_trace_oscillatory'):
    target=dest/folder;target.mkdir(exist_ok=True)
    for name in ['source_to_tdi.json','source_to_tdi.png','onset_sensitivity.json','onset_sensitivity.png']:
        if (HERE/folder/name).exists():shutil.copy2(HERE/folder/name,target/name)
for file in (ROOT/'prd_coherent').glob('*.py'):compile(file.read_text(),str(file),'exec')
ignore=ROOT/'.gitignore';text=ignore.read_text();entry='/FillingGap/results_prd_coherent_20261015/'
if entry not in text:ignore.write_text(text.rstrip()+'\n\n'+entry+'\n')
print(json.dumps(dict(archive=str(dest),source_manifest_files=len(json.loads((OUT/'runtime_source_manifest.json').read_text())),current_status=json.loads((OUT/'status.json').read_text())),indent=2))
