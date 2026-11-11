"""Table 1: exact numbers, original 6 seasons, nominal plant. Writes table1.tex (for the paper) and table1_preview.tex
(standalone, compiled to a PDF for viewing). Source: numbers.json T4 only."""
import statistics as S
from common import *

T4 = NUM["T4 real prices"]
def ms(xs, nd):
    return f"{S.mean(xs):.{nd}f}" if len(xs) == 1 else f"{S.mean(xs):.{nd}f} $\\pm$ {S.stdev(xs):.{nd}f}"

lines = []
for price, arm, head in (("AGC f2.5", "agc25", "\\textit{\\euro2.50 fruit, challenge energy prices}"), ("NL f1.6", "nl16", "\\textit{\\euro1.60 fruit, Dutch energy prices}")):
    lines.append(f"\\multicolumn{{4}}{{l}}{{{head}}} \\\\")
    for name, rs in (("Rules", [T4["agc-rules raw mm0"]]), ("Rules + filter", [T4["agc-rules +filter mm0"]]),
                     ("RL", [T4[f"d0.01_s{s}_{arm} raw mm0"] for s in range(3)]), ("RL + filter", [T4[f"d0.01_s{s}_{arm} +filter mm0"] for s in range(3)])):
        row = [ms([r["hours"] for r in rs], 1), ms([r["yield_kg"] for r in rs], 2), ms([r[price] for r in rs], 2)]
        lines.append(f"\\quad {name} & " + " & ".join(row) + " \\\\")
    lines.append("\\midrule")
lines = lines[:-1]
body = "\n".join(lines)
tab = f"""\\begin{{table}}[t]
\\centering\\small
\\begin{{tabular}}{{lrrr}}
\\toprule
Controller & Violation h & Yield (kg/m$^2$) & Profit (\\euro/m$^2$) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\caption{{Per 60-day season, original 6 seasons, nominal plant. RL agents are trained at the prices they are scored at; RL rows are mean $\\pm$ SD over 3 seeds.}}
\\label{{tab:main}}
\\end{{table}}
"""
OUT.mkdir(exist_ok=True); (OUT/"table1.tex").write_text(tab)
prev = ("\\documentclass[border=6pt]{standalone}\n\\usepackage{booktabs,eurosym}\n\\usepackage[T1]{fontenc}\n\\begin{document}\n\\small\\begin{tabular}{lrrr}\n\\toprule\n"
        "Controller & Violation h & Yield (kg/m$^2$) & Profit (\\euro/m$^2$) \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{document}\n")
(OUT/"table1_preview.tex").write_text(prev)
print(body)
