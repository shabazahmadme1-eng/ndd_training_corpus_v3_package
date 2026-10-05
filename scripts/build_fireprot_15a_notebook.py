"""Build the Tier 1.5a FireProt Colab notebook from the reviewed cell source.

Single source of truth is scripts/colab_fireprot_15a.py; this wraps it in one
code cell plus a markdown header so the notebook never drifts from review.
Writes notebooks/fireprot_15a.ipynb and validates it with nbformat.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = (ROOT / "scripts/colab_fireprot_15a.py").read_text()
    compile(source, "colab_fireprot_15a.py", "exec")
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Tier 1.5a — FireProt homologue-free benchmark\n"
            "\n"
            "Eval-only, GPU-light (80 proteins, 2427 rows). Upload "
            "`dist/fireprot_15a_bundle.zip` to `MyDrive/MIPO_NDD/` first, select a "
            "Colab GPU runtime, then run the cell below. It pins the v3 project "
            "extraction, writes in the tested scoring modules, asserts the decided "
            "post-exclusion corpus shape, extracts FireProt features, and scores "
            "every v1 champion checkpoint with `--allow-probeless`. Results land in "
            "`MyDrive/MIPO_NDD/fireprot_15a_results/`. See `TIER15_DESIGN.md` "
            "amendments for the dedupe decision and the probe caveat."
        ),
        nbf.v4.new_code_cell(source),
    ]
    out = ROOT / "notebooks/fireprot_15a.ipynb"
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, str(out))
    nbf.validate(nbf.read(str(out), as_version=4))
    print("wrote + validated", out)


if __name__ == "__main__":
    main()
