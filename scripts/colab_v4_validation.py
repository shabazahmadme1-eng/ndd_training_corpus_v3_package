# MIPO-NDD: paste this entire file into ONE Colab cell, or use the companion notebook.
STORAGE = 'local'                      # local: upload to /content; drive: mount Google Drive
SCOPE = 'mechanism'                    # mechanism (10 genes), direct, or curriculum
MODEL_SIZE = '650M'
USE_LLR = True
TEST_GENES = None                      # None: fixed 10-gene mechanism panel; direct/curriculum: 3-gene pilot
SEEDS = [42, 123, 2026]                # Training variability; seeds are not independent biological samples
SPLIT_SEED = 42                        # Same validation/calibration split across model seeds.
VALIDATION_GROUPS = 5
CALIBRATION_GROUPS = 3
STEPS_PER_EPOCH = 3000
EPOCHS = 20                            # Direct mode only; curriculum uses its stage budgets.
PATIENCE = 6
DO_FEATURES = None                     # None/True: resume and fill missing sites; False: require complete cache
DO_TRAIN = True
DO_BASELINE = True
DO_ABLATIONS = True
ABLATIONS = 'esm_mlp,no_structure,no_global_kernel'  # Matched component controls; one seed is a pilot only

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
import pandas as pd
from IPython.display import display

assert STORAGE in {'local', 'drive'}
if STORAGE == 'drive':
    from google.colab import drive
    drive.mount('/content/drive')
    STORAGE_ROOT = Path('/content/drive/MyDrive/MIPO_NDD')
else:
    STORAGE_ROOT = Path('/content')
ARCHIVE = STORAGE_ROOT/'mipo_ndd_v4_colab_upload.zip'
assert ARCHIVE.exists(), f'Upload the updated release ZIP to {ARCHIVE}'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def banner(text):
    print('\n'+'='*78+'\n'+text+'\n'+'='*78, flush=True)

def run_script(script, *args):
    subprocess.run([sys.executable, str(PROJECT/'scripts'/script), *map(str, args)], check=True)

def run(*args):
    subprocess.run([sys.executable, '-m', 'mipo', *map(str, args)], check=True)

banner('1. INSTALL UPDATED RELEASE')
archive_hash = sha(ARCHIVE)
PROJECT = Path('/content')/f'mipo_ndd_v4_validation_{archive_hash[:12]}'
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as archive:
    for item in archive.infolist():
        target = (PROJECT/item.filename).resolve()
        if target != PROJECT and PROJECT not in target.parents:
            raise ValueError('Unsafe archive path')
    archive.extractall(PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[test]'], check=True)
import torch
assert torch.cuda.is_available(), 'Choose a GPU runtime'
print('GPU:', torch.cuda.get_device_name(0))
print('Release SHA256:', archive_hash)
assert SCOPE in {'mechanism', 'direct', 'curriculum'}
if TEST_GENES is None:
    TEST_GENES = (pd.read_csv(PROJECT/'v4/robust/mechanism_panel.csv').gene.tolist()
                  if SCOPE == 'mechanism' else ['ASPA', 'KCNQ2', 'PAX6'])
assert MODEL_SIZE in {'150M', '650M'}
assert SEEDS and TEST_GENES and len(set(SEEDS)) == len(SEEDS) and len(set(TEST_GENES)) == len(TEST_GENES)
assert VALIDATION_GROUPS >= 2 and CALIBRATION_GROUPS >= 1
assert STEPS_PER_EPOCH > 0 and EPOCHS > 0 and PATIENCE > 0

banner('2. FIX SPLITS AND CHECKPOINT SELECTION BEFORE TRAINING')
DATA_SCOPE = 'curriculum' if SCOPE == 'mechanism' else SCOPE
CORPUS = PROJECT/f'v4/robust/{DATA_SCOPE}.csv.gz'
RESOURCES = PROJECT/'v4/resources'
METADATA = PROJECT/'v4/robust/assay_metadata.csv'
SOURCE_SPLITS = PROJECT/f'v4/robust/{SCOPE}_splits.json'
print('Primary analysis excludes consensus and documented duplicate deposits; see v4/robust/build_report.json')
SPLITS = PROJECT/f'active_{SCOPE}_multigene_splits.json'
run_script('prepare_validation_splits.py', '--corpus', CORPUS, '--source', SOURCE_SPLITS,
           '--out', SPLITS, '--validation-groups', VALIDATION_GROUPS,
           '--calibration-groups', CALIBRATION_GROUPS, '--seed', SPLIT_SEED,
           '--report-test-genes', ','.join(TEST_GENES), '--resources', RESOURCES)
split = json.loads(SPLITS.read_text())
support = json.loads(SPLITS.with_suffix('.support.json').read_text())
identity = json.loads(SPLITS.with_suffix('.identity.json').read_text())
if identity['flagged']:
    print(f"WARNING: test genes within {identity['flag_threshold']:.0%} sequence identity of a train gene "
          f"(similarity diagnostic; assess homology-aware splits before interpreting generalization): {identity['flagged']}")
folds = sorted({i for gene in TEST_GENES for i, fold in enumerate(split['folds']) if gene in fold['test']})
found = {g for i in folds for g in split['folds'][i]['test']}
assert set(TEST_GENES) <= found, f'Test genes absent from outer folds: {set(TEST_GENES)-found}'
for i in folds:
    fold = split['folds'][i]
    print(f"Fold {i}: test={fold['test']}, validation={fold['validation']}, calibration={fold['calibration']}, train genes={len(fold['train'])}")
    counts = support[str(fold['fold'])]
    print('Number of genes per task in each role:')
    display(pd.DataFrame(counts).T)
    for task, roles in counts.items():
        if roles['test'] and (not roles['validation'] or not roles['calibration']):
            print(f'Task coverage gap: {task}. Validation/calibration transfer for this task is not established.')
# Joint training for the mixed NDD/non-NDD mechanism panel; NDD specialization would
# change the target population. The measured broad corpus remains gene-held-out.
CONFIG_SCOPE = 'direct' if SCOPE == 'mechanism' else SCOPE
config = json.loads((PROJECT/f'configs/v4_robust_{CONFIG_SCOPE}.json').read_text())
config.update(seed=SEEDS[0], use_llr=USE_LLR, require_llr=USE_LLR,
              steps_per_epoch=STEPS_PER_EPOCH, patience=PATIENCE,
              selection_metric='macro_gene_spearman', min_validation_genes=VALIDATION_GROUPS)
if 'stages' in config:
    print('Curriculum stage budgets:', [(s['name'], s['epochs']) for s in config['stages']])
    max_epochs = sum(s['epochs'] for s in config['stages'])
else:
    config['epochs'] = EPOCHS
    max_epochs = EPOCHS
CONFIG = PROJECT/'active_v4_validation.json'
CONFIG.write_text(json.dumps(config, indent=2))
models = list(dict.fromkeys((['mipo'] if DO_TRAIN else []) +
                           ([x.strip() for x in ABLATIONS.split(',') if x.strip()] if DO_ABLATIONS else [])))
protocol = {'release_sha256': archive_hash, 'corpus_sha256': sha(CORPUS),
            'split_sha256': sha(SPLITS), 'metadata_sha256': sha(METADATA),
            'config': config, 'model_size': MODEL_SIZE, 'seeds': SEEDS, 'folds': folds, 'models': models, 'release_policy': 'robust_measured_v1'}
protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:16]
EXPERIMENT = STORAGE_ROOT/f'v4_validation_{SCOPE}_{MODEL_SIZE}_{protocol_hash}'
EXPERIMENT.mkdir(parents=True, exist_ok=True)
for source, name in [(CONFIG, 'config.json'), (SPLITS, 'splits.json')]:
    shutil.copy2(source, EXPERIMENT/name)
(EXPERIMENT/'protocol.json').write_text(json.dumps(protocol, indent=2))
RUNS = EXPERIMENT/'runs'
print('Outputs:', EXPERIMENT)
print(f'{len(models)*len(SEEDS)*len(folds)} requested model/seed/fold runs: {models}')
print(f"{STEPS_PER_EPOCH} batches x {config['batch_size']} = {STEPS_PER_EPOCH*config['batch_size']:,} sample draws/epoch; max {max_epochs} epochs; patience {PATIENCE}")
print('Selection: mean assay Spearman within each validation gene, then equal mean across genes.')
print('Expanded holdouts change training data; compare models within this new protocol.')

banner('3. REUSE OR EXTRACT FROZEN FEATURES')
MODEL = {'150M': 'facebook/esm2_t30_150M_UR50D', '650M': 'facebook/esm2_t33_650M_UR50D'}[MODEL_SIZE]
FEATURE_TAG = f'v4_{MODEL_SIZE}_{"masked" if USE_LLR else "no_llr"}'
CACHE = STORAGE_ROOT/FEATURE_TAG
LOCAL_CACHE = CACHE if STORAGE == 'local' else PROJECT/FEATURE_TAG
FEATURE_ARCHIVE = STORAGE_ROOT/f'{FEATURE_TAG}.zip'
if not (CACHE/'manifest.json').exists() and FEATURE_ARCHIVE.exists():
    with zipfile.ZipFile(FEATURE_ARCHIVE) as archive:
        names = set(archive.namelist())
        if f'{FEATURE_TAG}/manifest.json' in names:
            destination = STORAGE_ROOT
        elif 'manifest.json' in names:
            destination = CACHE
        else:
            raise ValueError(f'{FEATURE_ARCHIVE} must contain manifest.json at its root or inside {FEATURE_TAG}/')
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
        for item in archive.infolist():
            target = (destination/item.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError('Unsafe feature archive path')
        archive.extractall(destination)
    print('Uploaded feature cache extracted:', CACHE)
# Reuse embeddings but fill newly measured sites when an existing cache is supplied.
extract_features = DO_FEATURES is not False
if extract_features:
    print('Building/resuming ESM embeddings and masked LLR. This is a separate GPU stage before training.')
    args = ['features', '--corpus', CORPUS, '--resources', RESOURCES, '--cache', CACHE, '--model', MODEL]
    if not USE_LLR:
        args.append('--no-llr')
    run(*args)
assert (CACHE/'manifest.json').exists(), f'Feature cache missing: {CACHE}; set DO_FEATURES=True'
if CACHE.resolve() != LOCAL_CACHE.resolve():
    shutil.copytree(CACHE, LOCAL_CACHE, dirs_exist_ok=True)
print('Local cache ready:', LOCAL_CACHE)

banner('4. TRAIN AND RUN MATCHED ABLATIONS')
if models:
    started = time.time()
    run_script('run_experiments.py', '--corpus', CORPUS, '--resources', RESOURCES,
               '--cache', LOCAL_CACHE, '--splits', SPLITS, '--config', CONFIG,
               '--metadata', METADATA, '--out', RUNS, '--folds', ','.join(map(str, folds)),
               '--seeds', ','.join(map(str, SEEDS)), '--ablations', ','.join(models))
    print(f'Run time: {(time.time()-started)/60:.1f} min')

banner('5. TRAINING HISTORY AND VALIDATION DIAGNOSTICS')
selected_runs = []
for path in sorted(RUNS.glob('*/seed_*/fold_*/history.json')):
    if int(path.parent.name.split('_')[1]) not in folds or int(path.parent.parent.name.split('_')[1]) not in SEEDS:
        continue
    selected_runs.append(path.parent)
    print('\n', path.parent.relative_to(RUNS))
    history = pd.DataFrame(json.loads(path.read_text()))
    losses = pd.json_normalize(history.loss_components).add_prefix('train_')
    display(pd.concat([history[['epoch', 'loss', 'validation_spearman', 'lr', 'bad_epochs']], losses], axis=1).round(5))
    status = json.loads((path.parent/'training_status.json').read_text())
    print('Saved checkpoint / stopping:', status)
    per_gene = pd.DataFrame(history.validation_per_gene.tolist(), index=history.epoch)
    display(per_gene.round(4))
    print('Best checkpoint validation by gene and assay:')
    validation = pd.read_csv(path.parent/'validation_metrics.csv')
    display(validation[['gene', 'task', 'n', 'spearman', 'bias', 'mean_sigma', 'coverage_90_gaussian']].round(4))
    print('The history describes this validation panel; it does not establish the cause of a plateau.')

if DO_BASELINE and USE_LLR:
    banner('6. EXACT-ROW CACHED LLR BASELINE')
    run_script('compare_cached_llr.py', '--corpus', CORPUS, '--resources', RESOURCES,
               '--cache', LOCAL_CACHE, '--runs', RUNS)

banner('7. TEST RANKING AND INTERVAL DIAGNOSTICS')
if SCOPE == 'mechanism' and selected_runs:
    run_script('evaluate_property_contrasts.py', '--runs', RUNS,
               '--out', EXPERIMENT/'paired_property_contrasts.csv')
    print('Paired rank contrasts are descriptive assay-separation diagnostics, not causal folding labels.')
comparisons, intervals = [], []
for folder in selected_runs:
    model = folder.parents[1].name
    seed = int(folder.parent.name.split('_')[1])
    fold = int(folder.name.split('_')[1])
    metrics_path = folder/'test_metrics.csv'
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, 'model', model)
        metrics.insert(1, 'seed', seed)
        metrics.insert(2, 'fold', fold)
        intervals.append(metrics)
    comparison = folder/'matched_llr_metrics.csv'
    if comparison.exists():
        rows = pd.read_csv(comparison)
        rows.insert(0, 'model', model)
        rows.insert(1, 'seed', seed)
        rows.insert(2, 'fold', fold)
        comparisons.append(rows)
if comparisons:
    table = pd.concat(comparisons, ignore_index=True)
    if SCOPE == 'mechanism':
        panel = pd.read_csv(PROJECT/'v4/robust/mechanism_panel.csv').set_index('gene')
        table['evaluation_part'] = table.gene.map(panel.part)
        table['previously_examined'] = table.gene.map(panel.previously_examined)
    else:
        table['evaluation_part'] = 'ndd_evaluation'
    table.to_csv(EXPERIMENT/'matched_comparison.csv', index=False)
    display(table.round(4))
    if table[['model_spearman', 'llr_spearman']].notna().all().all():
        gene_scores = table.groupby(['model', 'seed', 'evaluation_part', 'gene'])[['model_spearman', 'llr_spearman', 'delta']].mean()
        per_seed = gene_scores.groupby(['model', 'seed', 'evaluation_part']).mean()
        print('Macro across genes, separately for each seed and NDD/non-NDD panel (all assays retained):')
        display(per_seed.round(4))
        per_seed.to_csv(EXPERIMENT/'macro_comparison_by_seed.csv')
        if len(SEEDS) > 1:
            print('Mean and standard deviation across seeds; this is training variability, not uncertainty across genes:')
            display(per_seed.groupby(['model', 'evaluation_part']).agg(['mean', 'std']).round(4))
    else:
        print('Some correlations are undefined: inspect the assay table; headline macro comparison suppressed.')
if intervals:
    table = pd.concat(intervals, ignore_index=True)
    table.to_csv(EXPERIMENT/'test_diagnostics.csv', index=False)
    display(table[['model', 'seed', 'gene', 'task', 'n', 'spearman', 'bias', 'mean_sigma',
                   'coverage_90_gaussian', 'coverage_calibrated', 'calibrated_interval_rows']].round(4))
    print('Gaussian coverage uses raw predicted sigma. Calibrated coverage uses separate calibration groups.')
    print('Missing calibrated coverage means no supported calibration interval. Nominal coverage is 0.90; transfer to unseen genes is empirical.')
    print('Bias, sigma, RMSE and NLL are in training-task-transformed units. Raw-score predictions and intervals are also saved.')
print(f'Evaluated {len(folds)} outer folds and {len(SEEDS)} seeds. Test results must not select checkpoints or tune this protocol.')
banner(f'DONE: {EXPERIMENT}')
if STORAGE == 'local':
    print('Download results before this runtime is deleted. Use the notebook download cell, or zip EXPERIMENT yourself.')
