# Contrast pilot cells (paste into Colab; not stored in the .ipynb)

Pilot: mipo vs esm_mlp, seed 42, folds testing ASPA/KCNQ2/PAX6,
contrast_weight=0.2, contrast_fraction=0.10. Own folder
`MyDrive/MIPO_NDD/contrast_pilot_v1`; never touches the benchmark grid.
Read validation numbers only; mean_contrast_loss must be > 0.

## Cell 10 — run the pilot in parallel (2 folds; resumable; refuses double-launch)

```python
# 10. CONTRAST PILOT (own folder; never touches the benchmark grid).
import json, subprocess, sys
from pathlib import Path

ARCHIVE = Path('/content/drive/MyDrive/MIPO_NDD/mipo_ndd_v4_colab_upload.zip')
h = hashlib.sha256()
with open(ARCHIVE, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
PROJECT = Path('/content')/('mipo_ndd_v4_validation_' + h.hexdigest()[:12])
CORPUS = PROJECT/'v4/robust/curriculum.csv.gz'
RESOURCES = PROJECT/'v4/resources'
SPLITS = PROJECT/'active_curriculum_multigene_splits.json'
METADATA = PROJECT/'v4/robust/assay_metadata.csv'
LOCAL_CACHE = PROJECT/'v4_650M_masked'
assert CORPUS.exists() and RESOURCES.exists() and SPLITS.exists() and METADATA.exists() and (LOCAL_CACHE/'manifest.json').exists()
PILOT = Path('/content/drive/MyDrive/MIPO_NDD/contrast_pilot_v1')
LOGS = PILOT/'logs'
LOGS.mkdir(exist_ok=True)
old = subprocess.run(['pgrep', '-af', 'run_experiments'], capture_output=True, text=True).stdout.strip()
assert not old, 'jobs already running, not launching again:\n' + old
config = json.loads((PROJECT/'configs/v4_robust_curriculum.json').read_text())
config.update(seed=42, use_llr=True, require_llr=True, steps_per_epoch=3000, patience=6, selection_metric='macro_gene_spearman', min_validation_genes=5, contrast_weight=0.2, contrast_fraction=0.10)
config['amp'] = False
cfg = PROJECT/'active_pilot_config.json'
cfg.write_text(json.dumps(config, indent=2))
BASE = [sys.executable, str(PROJECT/'scripts/run_experiments.py'), '--corpus', str(CORPUS), '--resources', str(RESOURCES), '--cache', str(LOCAL_CACHE), '--splits', str(SPLITS), '--config', str(cfg), '--metadata', str(METADATA), '--out', str(PILOT/'runs'), '--seeds', '42']
log1 = open(LOGS/'mipo_fold57.log', 'a')
p1 = subprocess.Popen(BASE + ['--folds', '57', '--ablations', 'mipo'], stdout=log1, stderr=subprocess.STDOUT, start_new_session=True)
print('mipo fold57 pid', p1.pid)
log2 = open(LOGS/'esm_mlp_fold12.log', 'a')
p2 = subprocess.Popen(BASE + ['--folds', '12', '--ablations', 'esm_mlp'], stdout=log2, stderr=subprocess.STDOUT, start_new_session=True)
print('esm_mlp fold12 pid', p2.pid)
log3 = open(LOGS/'esm_mlp_fold57.log', 'a')
p3 = subprocess.Popen(BASE + ['--folds', '57', '--ablations', 'esm_mlp'], stdout=log3, stderr=subprocess.STDOUT, start_new_session=True)
print('esm_mlp fold57 pid', p3.pid)
```

## Cell 11 — readout (validation only)

```python
# 11. PILOT READOUT (validation only; mean_contrast_loss must be > 0).
import json
from pathlib import Path
import pandas as pd

PILOT = Path('/content/drive/MyDrive/MIPO_NDD/contrast_pilot_v1')
rows = []
for hist in sorted((PILOT/'runs').glob('*/seed_*/fold_*/history.json')):
    model, fold = hist.parts[-4], hist.parts[-2]
    h = pd.DataFrame(json.loads(hist.read_text()))
    lc = pd.json_normalize(h.loss_components)
    val = pd.read_csv(hist.parent/'validation_metrics.csv')
    rows.append({'model': model, 'fold': fold, 'epochs': len(h),
                 'mean_contrast_loss': lc['contrast'].mean(),
                 'best_val_spearman': h.validation_spearman.max(),
                 'val_macro_spearman': val.spearman.mean()})
out = pd.DataFrame(rows).sort_values(['fold', 'model'])
print(out.round(4).to_string(index=False))
print('\nRule: mean_contrast_loss must be > 0 (objective alive); '
      'compare best_val_spearman mipo vs esm_mlp per fold.')
```

## Cell 12 — field wake-up check (same recipe as grid diagnosis B)

```python
# 12. FIELD WAKE-UP CHECK on pilot best checkpoints (mipo + esm_mlp, both folds).
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mipo.train import restore, loader
from mipo.common import to_device
from mipo.data import FeatureStore, TargetTransform, Metadata
from mipo.corpus import read_corpus
cands = sorted(Path('/content').glob('mipo_ndd_v4_validation_*'))
PROJECT = max(cands, key=lambda p: sum(1 for _ in p.glob('active_pilot_config.json')))
CORPUS = PROJECT/'v4/robust/curriculum.csv.gz'
RESOURCES = PROJECT/'v4/resources'
METADATA = PROJECT/'v4/robust/assay_metadata.csv'
LOCAL_CACHE = PROJECT/'v4_650M_masked'
PILOT = Path('/content/drive/MyDrive/MIPO_NDD/contrast_pilot_v1')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
corpus, store = read_corpus(str(CORPUS)), FeatureStore(str(RESOURCES), str(LOCAL_CACHE))
PIECES = ['local (seq window + field at site)', 'field pool (whole-protein field)', 'global context', 'mutation impulse + LLR']
records = []
for model_name in ['mipo', 'esm_mlp']:
    ckpts = {}
    for p in (PILOT/'runs').glob(f'{model_name}/seed_42/fold_*/best.pt'):
        if (p.parent/'test_summary.json').exists():
            ckpts[int(p.parent.name.split('_')[1])] = p
    print(model_name, 'checkpoints:', sorted(ckpts), flush=True)
    for fold, ckpt in sorted(ckpts.items()):
        model, state = restore(str(ckpt), device)
        model.eval()
        prov = state['provenance']
        config = dict(prov['config'], batch_size=64)
        rows = corpus[corpus.gene.isin(prov['fold']['test'])]
        rows = rows.groupby('gene', group_keys=False).apply(lambda g: g.sample(min(len(g), 2000), random_state=0))
        dl = loader(rows, store, TargetTransform(state['transform']), Metadata(str(METADATA), state['metadata_vocab']), config)
        gene_of = dict(zip(dl.dataset.rows.row_id, dl.dataset.rows.gene))
        captured, g_all, s_all, genes = [], [], [], []
        hook = model.fusion.register_forward_pre_hook(lambda m, inp: captured.append(inp[0].detach()))
        with torch.inference_mode():
            for b in dl:
                ids = torch.as_tensor(b['row_id']).cpu().numpy()
                out = model(to_device(b, device))
                x = captured.pop().float().reshape(len(ids), 4, -1).norm(dim=-1)
                s_all.append((x/x.sum(-1, keepdim=True)).cpu().numpy())
                g_all.append(out['gates'].float().cpu().numpy())
                genes += [gene_of[i] for i in ids]
        hook.remove()
        df = pd.DataFrame(np.hstack([np.vstack(g_all), np.vstack(s_all)]),
                          columns=[f'gate | {p}' for p in PIECES]+[f'share | {p}' for p in PIECES])
        df['gene'] = genes
        m = df.groupby('gene').mean().reset_index()
        m['model'], m['fold'] = model_name, fold
        records.append(m)
usage = pd.concat(records, ignore_index=True)
usage.to_csv(PILOT/'pilot_fusion_usage.csv', index=False)
print(usage.groupby('model')[[c for c in usage if '|' in c]].mean().T.round(4))
print('\nGrid baseline: mipo field-pool share was 0.007. Above that = the field is waking up.')
```

## Cell 13 — pilot v2 launcher (2 arms x 2 folds, 4 parallel procs)

Prereq: upload `dist/mipo_ndd_v4_colab_upload_staged_v2.zip` as
`mipo_ndd_v4_colab_upload.zip` on Drive, re-run cell 2 with the same
features-only settings (fast: everything cached). Arm A = modality_dropout 0
(config-only); Arm B = switch gate init + balance loss 1e-2. Baseline = pilot
v1 mipo runs. Separate out dirs per arm (config guard requires it).

```python
import hashlib, json, subprocess, sys
from pathlib import Path
ARCHIVE = Path('/content/drive/MyDrive/MIPO_NDD/mipo_ndd_v4_colab_upload.zip')
h = hashlib.sha256()
with open(ARCHIVE, 'rb') as f:
    for chunk in iter(lambda: f.read(1 << 20), b''):
        h.update(chunk)
PROJECT = Path('/content')/('mipo_ndd_v4_validation_' + h.hexdigest()[:12])
CORPUS = PROJECT/'v4/robust/curriculum.csv.gz'
RESOURCES = PROJECT/'v4/resources'
SPLITS = PROJECT/'active_curriculum_multigene_splits.json'
METADATA = PROJECT/'v4/robust/assay_metadata.csv'
LOCAL_CACHE = PROJECT/'v4_650M_masked'
assert CORPUS.exists() and RESOURCES.exists() and SPLITS.exists() and METADATA.exists() and (LOCAL_CACHE/'manifest.json').exists()
PILOT = Path('/content/drive/MyDrive/MIPO_NDD/contrast_pilot_v2')
LOGS = PILOT/'logs'
LOGS.mkdir(exist_ok=True)
old = subprocess.run(['pgrep', '-af', 'run_experiments'], capture_output=True, text=True).stdout.strip()
assert not old, 'jobs already running, not launching again:\n' + old
base = json.loads((PROJECT/'configs/v4_robust_curriculum.json').read_text())
base.update(seed=42, use_llr=True, require_llr=True, steps_per_epoch=3000, patience=6, selection_metric='macro_gene_spearman', min_validation_genes=5, contrast_weight=0.2, contrast_fraction=0.10)
base['amp'] = False
cfgs = {}
for arm, extra in [('A', {'modality_dropout': 0}), ('B', {'fusion_gate_init': 'switch', 'balance_weight': 0.01})]:
    c = dict(base, **extra)
    p = PROJECT/f'active_pilot_v2_{arm}.json'
    p.write_text(json.dumps(c, indent=2))
    cfgs[arm] = p
jobs = [('A', '12'), ('A', '57'), ('B', '12'), ('B', '57')]
for arm, fold in jobs:
    cmd = [sys.executable, str(PROJECT/'scripts/run_experiments.py'), '--corpus', str(CORPUS), '--resources', str(RESOURCES), '--cache', str(LOCAL_CACHE), '--splits', str(SPLITS), '--config', str(cfgs[arm]), '--metadata', str(METADATA), '--out', str(PILOT/f'runs_{arm}'), '--seeds', '42', '--folds', fold, '--ablations', 'mipo']
    log = open(LOGS/f'arm{arm}_fold{fold}.log', 'a')
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(f'arm {arm} fold {fold} pid', proc.pid)
```
