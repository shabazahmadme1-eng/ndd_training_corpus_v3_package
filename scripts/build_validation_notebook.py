"""Build the one-cell Colab notebook from the maintained Python entry point."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
source = (root/'scripts/colab_v4_validation.py').read_text(encoding='utf-8')
compile(source, 'colab_v4_validation.py', 'exec')
notebook = nbf.v4.new_notebook(cells=[
    nbf.v4.new_markdown_cell('''# MIPO-NDD: multi-gene validation

Select a GPU runtime, then upload `mipo_ndd_v4_colab_upload.zip` through the
Files sidebar to `/content/`. No Drive mount is needed. Run the cell below.
The release contains code, data and reference resources, **not precomputed ESM features**.
If you also upload your existing `v4_650M_masked.zip`, it is extracted and reused.
Otherwise features and masked LLR are computed first. Defaults:
The restored ten-gene mechanism panel (including PTEN and KRAS), five validation
groups, three separate calibration groups, seeds [42, 123, 2026], and four matched
models: **120 training runs**, plus feature extraction. This uses the measured broad
corpus with joint training. Eight panel genes are not NDD-verified; results are
reported separately. PTEN is a previously examined development case. Seeds measure
training variability, not independent biological replication. Shared-variant paired
rank contrasts diagnose assay separation; they do not prove folding mechanisms.
Set SCOPE='direct' for the smaller ASPA/KCNQ2/PAX6 pilot. Freeze an untouched
gene/family panel before making confirmatory generalization claims.
The primary views exclude consensus and duplicate deposits and add 11 reviewed
functional score sets. Calibration intervals require at least three calibration
genes for a task and can therefore be unavailable. This is not a coverage guarantee.

The script preserves outer tests and cluster boundaries, records each role,
selects checkpoints only on validation macro gene Spearman, and reports
Gaussian and calibrated interval coverage separately. New outputs have a
protocol-specific folder. Download results before the temporary runtime is deleted.
See `ROBUST_RELEASE.md` and `v4/robust/build_report.json`. Old training checkpoints
are incompatible; sequence-compatible feature caches are reused and missing sites filled.
'''),
    nbf.v4.new_code_cell(source),
    nbf.v4.new_markdown_cell('''## Download results

Run this after training. It can also save completed checkpoints after an interrupted
training cell, while the runtime is still alive. Feature caches are separate from this ZIP.
'''),
    nbf.v4.new_code_cell('''from google.colab import files
import shutil
from pathlib import Path
assert 'EXPERIMENT' in globals() and EXPERIMENT.exists(), 'Start the training cell first'
result_zip = shutil.make_archive(str(Path('/content') / (EXPERIMENT.name + '_results')), 'zip',
                                 root_dir=EXPERIMENT.parent, base_dir=EXPERIMENT.name)
files.download(result_zip)
''')])
notebook.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                     'language_info': {'name': 'python', 'version': '3.10'}, 'accelerator': 'GPU'}
nbf.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == 'code':
        compile(cell.source, 'colab_cell', 'exec')
for name in ['MIPO_NDD_V4_Upload_Colab.ipynb', 'MIPO_NDD_V4_Validation_Colab.ipynb', 'MIPO_NDD_V4_Colab.ipynb']:
    path = root/'notebooks'/name
    nbf.write(notebook, path)
    print(path)
