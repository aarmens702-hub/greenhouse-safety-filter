"""Shared style and data access for the paper figures. Reads only results/numbers.json and raw result files whose
sha256 is listed in docs/FROZEN.md (checked on every load). Figures are written to figures/out/."""
import hashlib, json, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT/"figures"; OUT = FIG/"out"; DATA = FIG/"data"
RES = ROOT/"results"; FR = RES/"filter"
NUM = json.loads((RES/"numbers.json").read_text())
SE = [("Amsterdam", 45), ("Amsterdam", 59), ("Amsterdam", 75), ("Wag", 5), ("Wag", 45), ("Wag", 85)]

# Okabe-Ito, colour-blind safe; the same controller keeps the same colour in every figure
C = {"rules": "#9A9A9A", "rules+filter": "#4D4D4D", "rl": "#56B4E9", "rl+filter": "#0072B2",
     "ams": "#E69F00", "wag": "#009E73", "accent": "#D55E00"}
COL1, COL2 = 3.3, 7.0     # AAAI one-column and full-width figure widths, inches

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

_FROZEN = {m.group(2): m.group(1) for m in re.finditer(r"^([0-9a-f]{64})\s+(\S+)$", (ROOT/"docs"/"FROZEN.md").read_text(), re.M)}

def rows(path):
    """Load a raw jsonl result file after checking its hash against FROZEN.md."""
    path = Path(path); h = hashlib.sha256(path.read_bytes()).hexdigest()
    key = next((k for k in _FROZEN if path.as_posix().endswith(k)), None)
    assert key and _FROZEN[key] == h, f"{path.name} is not the frozen version"
    return [json.loads(l) for l in open(path)]

hrs = lambda r: r["grade_rh"] + r["temp_h"] + r["co2_h"]

def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/f"{name}.pdf"); fig.savefig(OUT/f"{name}.png", dpi=220)
    plt.close(fig)
