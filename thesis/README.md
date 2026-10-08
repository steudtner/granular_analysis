# Diplomarbeit: Tabellen und neue Abbildungen

> Aktueller Stand nach Neufit beider Gitter: Für die Dokumentation die Dateien
> `kap7_aicc_aktuell.csv`, `kap7_1_tabellenzeilen_aktuell.tex` und
> `kap6_2_tabellenzeilen_aktuell.tex` verwenden. Vollständige Übergabe:
> `bilder/generiert/kap7_aicc_aktuell_dokumentationsbriefing.md`.
> Das unten beschriebene `run.sh` erzeugt weiterhin die historische Auswertung;
> die aktuellen Fit-Exporte werden mit `current_slope_comparison.py` aufbereitet.

Aus dem Workspace: `bash granular_analysis/thesis/run.sh`. Python kann über
`THESIS_PYTHON` überschrieben werden. Bestehende SVGs und LaTeX-Dateien werden nicht
geändert. Alle neuen Analyseartefakte liegen in `bilder/generiert/`.

## AICc und Datenstand

`kap7_aicc.csv`, `kap7_1_tabellenzeilen.tex`, `kap6_2_tabellenzeilen.tex` und
`kap7_aicc_bewertung.md` sind die gewünschten Tabellen/Interpretation, **mit Vorbehalt
zur historischen Rekonstruktion**. Tabellenspalten: Modell, D_eq/Delta, n, k, AIC, BIC,
AICc. Im LaTeX-Workspace erfolgt keine Eintragung.

Die archivierten R-CSV-Dateien enthalten weder n/k noch Fitobjekte. Aus dem dokumentierten
univariaten Modell folgt k=3+2b (b Knickpunkte); aus BIC-AIC=k(log(n)-2) folgt n.
Alle sechs Modelle liefern pro Gitter dieselbe ganze Zahl (42 bzw. 41), mit Rundungs-
abweichung <1e-7. Das ist eine Rekonstruktion, keine direkte Inspektion damaliger
`coef(fit)`. Die heutigen Profile haben 44/40 Zeilen und wurden später exportiert.
Sie werden nicht mit alten Kriterien kombiniert. Original-AIC/BIC bleiben unverändert.
Herkunftsdateien mit SHA256: `kap7_aicc_herkunft.json`.

Die Zuordnung Δ16/Δ20 folgt aus input.dat: primeJainSU 100/1600, primeHigParMidFluid
100/2000. D_eq=1 ist die nominelle Referenz im Auswertecode (individuelle Durchmesser
variieren). „1–5 Segmente“ in den Altdateien bedeutet 1–5 **Knickpunkte**, also 2–6
Geradenabschnitte; Linear ist bereits ein Geradenabschnitt.

Beide R-Skripte exportieren künftig n=nobs(fit), k=length(coef(fit))+1,
AIC(fit), AIC(fit,k=log(n)), AICc, außerdem `slope_fits.rds` und die tatsächlich
verwendeten endlichen Punkte in `fit_upper_edge.csv`. Kein Austausch der ML-Definition.
AICc ist bei n<=k+1 nicht definiert (R: NA; Tabellenroutine: ValueError).
Der Zufallsstart ist für künftige Fits fixiert. Fehlgeschlagene Fits werden gemeldet.
Die neue CSV-Routine fordert für die beiden vollständigen Tabellen sechs Modelle.

## Plot-Routinen und Eingaben

Aufruf mit PYTHONPATH=granular_analysis/src:
`python -m granular_analysis.thesis.plots ROUTINE DATEI [--zusatz DATEI] [--svg-fonttype path]`.
Fehlende Dateien erzeugen FileNotFoundError mit Dateipfad/Spalten; ungültige Daten
ValueError. Keine leeren Kurven oder Ersatzwerte. Jede Routine besitzt eine Docstring.
CSV-Trennzeichen Komma, Dezimalpunkt, `form` ist Kugel/Prolat/Oblat, `fall` eindeutiger
Fallname. Numerische Größen müssen endlich sein. Zeit in Sekunden; geometrische Größen
sind bereits durch dieselbe nominelle Referenz D_eq normiert.

| Routine | CSV-Spalten | Zweck / Stand |
|---|---|---|
| energie | form,fall,t,E_kin_star | Kap. 7.1; benötigt zusätzlich fall,t_qs,methode in `--zusatz` |
| translation | form,fall,t,E_trans_star | vorläufiger Vergleich ohne Endzeit; eine Abbildung je Prolat-Fall, jeweils mit Kugel und Oblat |
| runout | form,fall,t,L_star,H_star | Kap. 7.1/7.2; zeitaufgelöste Geometrie fehlt in aufbereiteter Form |
| alpha | form,fall,alpha,L_end_star,A_alpha_star | Kap. 6.2, genau 0.4/0.6/0.8; konsistente L_end fehlen |
| alpha_flaeche | form,fall,alpha,A_alpha_star | vorhandene Flächen für Kugel und primeCollNick, Teil des TODO |
| gitter | aufloesung,L_end_star,H_end_star,phi_z | Kap. 6.2; gemeinsame Endzustandsdefinition offen |
| kriterien | modell,aufloesung,n,k,AIC,BIC,AICc | Abb. 7.2, zwei Gitter separat |
| porositaet | form,fall,z_star,phi | Kap. 7.3; vorhandene letzte Snapshots, nicht als quasistatisch zertifiziert |
| skalierung | form,fall,chi,l,h | Kap. 7.2; Anfangsgeometrien/Skalierungsdatentabelle fehlen; Referenzen siehe LITERATUR.md |

Bei `skalierung` enthält `--zusatz` eine JSON-Liste mit `law` (Schlüssel aus SCALINGS),
`chi_ref`, `value_ref`, `source`. Nur belegte Referenzamplituden verwenden.
Zwei Gitter reichen für relative Differenzen, nicht für eine beobachtete Konvergenzordnung.
`phi_z` muss dieselbe dokumentierte Mittelung über dieselben Schnitte bezeichnen.

## Tatsächlich vorhandene Daten und Grenzen

Die Benutzerzuordnung ist in `prepare.py` implementiert: primeJainSU = Kugel,
oblaten/new-prime-gran-coll-2000_NewInput = Oblat; alle direkten Prolat-Simulationsfälle
mit input.dat, ohne old/. Kein Mittelwert über unterschiedliche Prolat-Fälle.
`anhang_dateninventar.json` führt vorhandene/fehlende Quelldateien und ihre Hashes auf.

Ältere Kugel-Energieexporte enthalten nur Translation, neue Formexporte Translation plus
Rotation. Die erzeugten Vorschauplots vergleichen deshalb ausschließlich die vorhandene
Spalte für Translationsenergie, E_trans/(rho_ref*g*D_eq^4). Der gewünschte Gesamtenergie-
vergleich mit quasistatischer Endmarkierung bleibt offen: dafür fehlen eine konsistente
Kugel-Gesamtenergie und die noch nicht festgelegte Endzeitdefinition. Es wird kein t_qs
erfunden. Fehlende Prolat-Exporte werden im Inventar ausgewiesen.

Die Alpha-Flächen stammen direkt aus alpha_sensitivity.csv. Die Geometrie-JSONs des
separaten Knickpunkt-Sweeps sind teilweise unvollständig bzw. haben abweichende Flächen;
daraus wird keine scheinbar konsistente L_end-Sensitivität zusammengestellt.
Die Porositätsprofile sind bestehende letzte Exporte aus slices/slice_summary.csv:
PRIME verwendet hier y als Vertikale, z als Spannweitenkoordinate. Diese Kurven sind
**Schnittprofile über z**, keine vertikalen Höhenprofile. Für ein vertikales Profil
müssen entsprechende horizontale Schichten neu ausgewertet werden. Endframes und
Partikelfilter der alten Exporte müssen vor dem Formvergleich fachlich abgeglichen werden.

## Font und SVG/LaTeX

`plotting/plotstyle.py`: Noto Sans 9 pt, Labels 9 pt, Legende 8 pt, Linien 1.2,
keine Gitter/Titel, Breiten 6.3/3.0 in, SVG-Text (`none`) und PDF. Legenden außerhalb
der Daten. `bbox=tight` kann die exportierte Außenbreite ändern; LaTeX setzt die
gewünschte Breite mit `width=\linewidth` bzw. `0.49\textwidth`.

Noto Sans wurde lokal in thesis/fonts bereitgestellt (Regular/Italic/Bold, SIL OFL).
Quelle: https://github.com/notofonts/noto-fonts/tree/main/hinted/ttf/NotoSans .
Die Routine registriert diese Fonts pro Prozess. Externe Nutzer können `configure`
mit `font_file` oder `THESIS_NOTO_FONT` aufrufen. Fehlt Noto beim Export, erfolgt ein
Fehler statt einer unbemerkten Ersatzschrift.

`svg.fonttype=none` erhält Text; es erzwingt **nicht allein** den Dokumentfont.
Bei gewöhnlichem Inkscape-Rendering muss Noto Sans dort installiert sein; bei
LaTeX-Textoverlay hängt die Textsetzung von der includesvg/inkscapelatex-Konfiguration
ab. Den fremden LaTeX-Workspace habe ich nicht untersucht oder verändert.
Windows ist hier nicht eingebunden. Auf dem Zielrechner Noto Sans in den Windows-
Schrifteinstellungen und in Inkscapes Schriftauswahl prüfen. Wenn dort nicht vorhanden,
`--svg-fonttype path` verwenden; diese Pfade werden mit dem lokal vorhandenen Noto
Sans erzeugt. Die Windows-Verfügbarkeit ist ausdrücklich nicht als geprüft markiert.
