# Testübersicht

Die Testsuite (252 Tests, ~85 s) prüft die Analyse-Pipelines gegen künstliche
Fixtures mit bekannten analytischen Antworten — keine echten
Simulationsdaten erforderlich.

## Ausführen

```bash
python -m pip install -e ".[dev]"
python -m pytest -q                    # testpaths aus pyproject.toml
```

Die GitHub-Actions-CI (`.github/workflows/ci.yml`) installiert TeX Live,
führt `ruff check` auf Fehler-Level aus und startet dieselbe Suite auf
`ubuntu-24.04` mit Python 3.9. Die zehn pyvista-abhängigen Renderer-Tests
werden in der CI übersprungen (`pytest.importorskip('pyvista')`; pyvista ist
ein optionales Extra und gehört nicht zum `dev`-Set).

## Testphilosophie

- **Synthetische Fixtures:** handgeschriebene Monitor-/Input-Dateien,
  analytische Kugel- und Ellipsoidanordnungen, gemockte Frames.
- **Analytische Validierung:** Algorithmen werden gegen geschlossene
  Referenzwerte geprüft (Kreis-/Ellipsenflächen, Corey Sf von Hand
  gerechnet, Wadell-Sphärizität gegen elementare Spheroid-Formeln,
  numerische Oberflächenintegrale per `dblquad`).
- **Konvergenznachweise:** Diskretisierungsabhängige Ergebnisse müssen gegen
  analytische Werte konvergieren (z. B. Porosität mit 10/72/720
  Konturpunkten).
- **Negativtests:** fehlerhafte Input-Zeilen, Monitore außerhalb des
  Input-Bereichs, Lücken in Zeitreihen, leere Slices und nichtfinale Werte
  müssen explizite Fehler oder NaNs erzeugen — niemals stillschweigend
  falsche Zahlen.

## Abdeckung nach Modul

| Testdatei | Tests | Inhalt |
| --- | --- | --- |
| `test_fluid_slice.py` | 37 | PRIME velocity/pressure/λ2/Q-Laden, Zellzentrierung, exakte Partikel/Fluid-Zeitpaare, Gitterzellen-Logik |
| `test_particle_shape.py` | 26 | Zingg-Ratios, Corey Sf, Wadell-Sphärizität, Sieblinie (Volumen/Masse/Anzahl), DIN EN ISO 14688-Klassen, CLI |
| `test_run_all.py` | 18 | Pipeline-Auswahl (`--only`/`--exclude`), Pfaderkennung, `--dry`, counter/time-Logik, Fehlercodes |
| `test_collapse.py` | 18 | TRN-Reader, Böschungskanten-Metrik, Quasistatik-Flag, t95-Kollapszeit, AICc-Modellwahl, Kontaktpaare, Zweigitter-Vergleich |
| `test_alignment.py` | 17 | Winkeldefinitionen (prolate a-Achse / oblate c-Achse), Gewichtung, isotrope Referenz, DBSCAN-Cluster |
| `test_slice_plotter.py` | 16 | Porositäts- und Flächenprofile, Alpha-Overlays, Legenden-/Farbskalenlayout |
| `test_deposit_orientation.py` | 16 | x-Bins, Winkelklassen, Achsenauswahl, Ausschlussregeln, Renderview-Kombination |
| `test_energy.py` | 14 | E = ½mv², Monitor-zu-Input-Zuordnung, Rotationsfilter (Share-Schwelle), Normalisierung |
| `test_porosity_six_spheres.py` | 9 | Alpha-Shape-Porosität gegen analytische Sechskugelanordnung, Konvergenz |
| `test_energy_smoothing.py` | 9 | Gleitender Mittelwert, Lücken, Fensterkanten, Punktreduktion |
| `test_particle_fluid.py` | 8 | 3D-Render + 2D-Fluidslice kombiniert, Grenzen, Farbskalen |
| `test_surface_metrics.py` | 7 | Länge/Höhe der Ablagerung aus Alpha-Formen |
| `test_renderview_video.py` | 7 | Frame-Auswahl, Wiedergabeplan, Kamera-Elevation, Zeitkonversion |
| `test_thesis.py` | 6 | Thesis-Plots und AICc-Berechnung |
| `test_render_export.py` | 6 | PDF/SVG-Export, Text als Vektorpfade, Vortex-Log-Farben |
| `test_porosity_time.py` | 6 | Zeitliche Profilauswahl, Mittelwert/Standardabweichung, Cache |
| `test_case_paths.py` | 6 | Auflösung von Case-Root/-results/-ellipsoid-Pfaden |
| `test_potential_reference.py` | 4 | Initial-Potentialenergie-Referenz aus Input-Koordinaten |
| `test_energy_comparison.py` | 4 | Case-Vergleich ohne Interpolation/Clipping |
| `test_alpha2d.py` | 4 | 2D-Alpha-Formen (Kreis, Rechteck, Randfälle) |
| `test_slope.py` | 3 | Piecewise-lineare Modelle, AIC/BIC-Score |
| `test_plotstyle.py` | 3 | Zentraler Stil (Locale, Exportgrößen) |
| `test_geometry_comparison.py` | 3 | Case-Vergleich der Geometrie-Pipelines |
| `test_monitor_history.py` | 2 | Priorität History-Datei vor Live-Monitor |
| `test_alpha_logging.py` | 2 | Alpha-Logging-Ausgaben |
| `test_state_manager.py` | 1 | Cache-Fingerprint/Invalidierung |

Die PyVista-abhängigen Tests in `test_particle_fluid.py`,
`test_render_export.py` und `test_renderview_video.py` laufen nur, wenn
pyvista installiert ist; andernfalls werden sie übersprungen.
