# Refactor-Protokoll — granular_analysis

Datum: 2026-06-03

## Durchgeführte Änderungen

### Neu: utils/porosity_utils.py
- `compute_porosity(V_outer, V_inner)` zentralisiert
- Vorher dreifach kopiert in `porosity2d.py`, `porosity3d.py`, `slice_depth_analysis.py`
- Alle drei Stellen importieren jetzt aus `utils.porosity_utils`

### Neu: utils/logging_utils.py
- `suppress_noise()` zentralisiert
- Vorher doppelt in `porosity3d_opentop.py` und `run_3d_opentop.py`

### Neu: utils/__init__.py
- Package hatte kein `__init__.py` — jetzt vorhanden mit re-exports

### geometry/alpha3d_variants.py
- Dupliziertes `_largest_polygon()` entfernt
- Importiert jetzt aus `alpha_utils.py`

### geometry/alpha_utils.py
- Bereinigte, vollständige Version mit Docstrings

### scripts/run_3d_opentop.py
- Lokales `suppress_noise()` entfernt, zentrales importiert
- Überlange Zeilenumbrüche geglättet

### scripts/run_depth_slice_analysis.py
- Wildcard-Import `from granular_analysis.config import *` → explizite Imports

### scripts/run_energy_analysis.py
- Wildcard-Import `from granular_analysis.config import *` → explizite Imports

### scripts/run_all.py
- Lokales `suppress_noise()` entfernt, zentrales importiert

### tests/test_alpha2d.py
- Defekt: `test_rectangle()` war falsch eingerückt (inside `test_too_few_points()`)
- Jetzt korrekt als eigenständige Funktion auf Top-Level

### Nicht geändert
- `alpha3d_open_top.py` vs `alpha3d_opentop_pca.py` — logisch different (open-top Heuristik vs. Prism-Modell); Konsolidierung nur mit fachlichem Input sinnvoll
- `read_ellipsoid_data.py` — stabiler Legacy-Code, nicht angefasst
- `ellipseSlices/` Daten — unverändert
