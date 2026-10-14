"""Resume the frozen independent evaluation with its missing batch constant.

The training package, selected checkpoint, test seeds and masks are unchanged.
The frozen evaluator imports settings without BATCH. Supply the intended 128
inference batch at module scope, then execute the existing audited evaluator.
"""
import sys
import json
import fcntl
import time
from pathlib import Path

ROOT = Path('/home/ljq/code/RingdownGapFilling_PRD')
sys.path.insert(0, str(ROOT))
from prd_conditioned import final
from prd_conditioned.settings import OUT
from prd_physical.common import sha, write

with (OUT/'pipeline.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = json.loads((OUT/'selection_frozen.json').read_text())
    progress = json.loads((OUT/'fresh_test/progress.json').read_text())
    assert progress['completed'] == progress['total'] == 6000
    assert sha(OUT/'finetune/best.pth') == selected['selection']['checkpoint_sha256']
    final.BATCH = 128
    write(OUT/'evaluation_recovery.json', {
        'reason':'Missing BATCH constant in frozen final-evaluation module',
        'batch_size':128,'training_source_files_modified':False,
        'checkpoint_sha256':selected['selection']['checkpoint_sha256'],
        'existing_test_sources_reused':6000,'time':time.time(),
    })
    final.main()
