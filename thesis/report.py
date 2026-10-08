"""Liste aller neuen Dateien und Kapitel-TODOs aus den tatsächlichen Outputs erzeugen."""
from pathlib import Path

root=Path(__file__).resolve().parents[2]
out=root/'bilder/generiert'
report=out/'anhang_lieferumfang.md'
source_files=[
 'granular_analysis/src/granular_analysis/plotting/plotstyle.py',
 'granular_analysis/src/granular_analysis/thesis/__init__.py',
 'granular_analysis/src/granular_analysis/thesis/aicc.py',
 'granular_analysis/src/granular_analysis/thesis/plots.py',
 'granular_analysis/src/granular_analysis/thesis/prepare.py',
 'granular_analysis/tests/test_thesis.py',
 'granular_analysis/thesis/higparmidfluid.py',
 'granular_analysis/thesis/current_slope_comparison.py',
 'granular_analysis/thesis/run.sh', 'granular_analysis/thesis/report.py',
 'granular_analysis/thesis/README.md','granular_analysis/thesis/LITERATUR.md',
 'granular_analysis/thesis/fonts/NotoSans-Regular.ttf',
 'granular_analysis/thesis/fonts/NotoSans-Italic.ttf',
 'granular_analysis/thesis/fonts/NotoSans-Bold.ttf','granular_analysis/thesis/fonts/LICENSE',
]
lines=['# Lieferumfang und offene Kapitel-TODOs','',
 '| Kapitel | Stand |', '|---|---|',
 '| Tab. 7.1 / 6.2, Anhang A.5 | CSV + booktabs + 5-Satz-Bewertung erzeugt. Historische n/k rekonstruiert; Herkunft vor Eintragung prüfen. Modellwahl nicht pauschal robust. |',
 '| Kap. 7.1 Energie | Vier SVG/PDF-Vergleiche der Translation; Gesamtenergie und quasistatische Markierung offen. Fehlender Cut-Prolat-Export im Inventar. |',
 '| Kap. 7.1/7.2 L(t), H(t) | Routine vorhanden; konsistente zeitaufgelöste Geometrietabelle fehlt. |',
 '| Kap. 6.2 Alpha | Fläche für Kugel/primeCollNick erzeugt. L_end und weitere Formen offen. |',
 '| Gitterstudien-Fazit | Vergleichsroutine und relative Differenzen implementiert; vergleichbare Endwerttabelle fehlt. Keine Konvergenzordnung aus zwei Gittern. |',
 '| Abb. 7.2 | AIC/BIC/AICc-Vergleich je Gitter erzeugt. |',
 '| Kap. 7.3 | Vorhandene Porositäts-Schnittprofile geplottet; z ist Spannweite, nicht vertikale Höhe. Endzustand/Filter abgleichen. |',
 '| Skalierungen | Routine und Primärquellen-Exponenten vorhanden; DNS-Geometrietabelle und belegte Referenzamplituden offen. |',
 '| LaTeX/Fonts | Noto Sans lokal bereitgestellt, Text-/Pfadoption vorhanden; Windows/Inkscape nicht zugänglich. |','',
 '## Validierung','',
 '9 Python-Tests bestanden (neue Kriterien-/Fehlerfalltests und bestehende Slope-Tests). '
 'Die bestehende pytest-Warnkonfiguration musste mit `-o filterwarnings=` übersteuert werden, '
 'weil die installierte pyparsing-Version die konfigurierte Warnklasse nicht besitzt. '
 'Beide R-Skripte geparst; realer Prüffit mit 44 Punkten und sechs Modellen bestätigt '
 'nobs, length(coef)+1 und AIC/BIC gegen die gespeicherten Fitobjekte. '
 'Der Prüffit liegt ausschließlich unter /tmp/thesis-rcheck16. '
 'SVGs auf Textobjekte/Noto-Sans und zugehörige PDFs geprüft; Kriterien- und Profilplot visuell geprüft.',
 '', '## Erzeugte Analyseartefakte','']
for p in sorted(set(out.rglob('*'))|{report}):
 if p.is_file() or p==report: lines.append(f'- `{p.relative_to(root)}`')
lines += ['', '## Neue Programm-/Dokumentationsdateien','']
lines += ['- `'+p+'`' for p in source_files]
lines += ['', '## Ergänzte bestehende Skripte','',
 '- `granular_analysis/src/granular_analysis/scripts/slope_analysis.R`',
 '- `granular_analysis/src/granular_analysis/scripts/segmented_regression.R`',
 '', 'Bestehende Simulationsdaten/SVGs und der externe LaTeX-Workspace wurden nicht verändert.','']
if (out/'kap7_aicc_aktuell.csv').is_file():
 lines[2:2] = ['**Aktueller Neufit beider Gitter:** Für Tab. 7.1/6.2 die Dateien mit `_aktuell` verwenden. Low: n=44, AICc bevorzugt fünf Abschnitte (drei Abschnitte nur 1.291 darüber); Mid: n=40, AICc bevorzugt drei Abschnitte. BIC bevorzugt drei Abschnitte bei beiden Gittern. Details: kap7_aicc_aktuell_dokumentationsbriefing.md. Die historische Rekonstruktion weiter unten ist überholt.', '']
report.write_text('\n'.join(lines),encoding='utf-8')
