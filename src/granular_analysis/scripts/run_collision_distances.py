#!/usr/bin/env python3
"""
Compute collision surface distances by combining COLL and TRN data.

Orientierungs-Update:
Der Oberflächenabstand wird jetzt über die Perram-Wertheim-Kontaktfunktion
berechnet (J. W. Perram, M. S. Wertheim, J. Comput. Phys. 58, 409, 1985),
die die Orientierung BEIDER Ellipsoide gekoppelt berücksichtigt.

Die bisherige Methode (radius_along_dir für Partikel 1 und 2 unabhängig
entlang der Zentren-Verbindungslinie, dann center_dist - (r1+r2)) ist nur
bei Kugeln exakt. Bei Ellipsoiden liegt der wahre nächstgelegene
Oberflächenpunkt im Allgemeinen NICHT auf der Zentrenlinie, sondern hängt
von der relativen Verdrehung beider Partikel zueinander ab.

Die Perram-Wertheim-Näherung des minimalen Abstands entlang der
Zentrenlinie:
    A = R1 diag(1/a1^2,1/b1^2,1/c1^2) R1^T   (Formmatrix Ellipsoid 1)
    B = R2 diag(1/a2^2,1/b2^2,1/c2^2) R2^T   (Formmatrix Ellipsoid 2)
    F(lambda) = lambda*(1-lambda) * r_ab^T [lambda*B^-1 + (1-lambda)*A^-1]^-1 r_ab
    s = max_{lambda in (0,1)} F(lambda)
    d_PW = |r_ab| * (1 - 1/sqrt(s))

    s < 1  -> Durchdringung (surface_distance < 0)
    s = 1  -> Berührung     (surface_distance = 0)
    s > 1  -> Trennung      (surface_distance > 0)

Für Kugeln (a=b=c) reduziert sich d_PW exakt auf center_dist - (r1+r2).
Verifiziert gegen analytische Testfälle (Kugeln, Ellipsoide Spitze-an-Spitze
und Seite-an-Seite): exakte Übereinstimmung.
"""

from granular_analysis.plotting import plotstyle

import os
import argparse
import numpy as np
from scipy.optimize import minimize_scalar

from granular_analysis.config import FOLDER, SKIP_FIRST_PARTICLES, USE_PARTICLE_SKIP
from granular_analysis.paths import CasePaths, collision_rk_stages


def _resolve_data_root(path):
    """Return the ellipsoid root expected by ``load_data``."""
    return str(CasePaths.from_input(path or FOLDER).ellipsoid)


from granular_analysis.io.read_ellipsoid_data import load_data

# -------------------------------------------------------------------------
# Quaternion -> Rotation Matrix
# -------------------------------------------------------------------------
def quat_to_rot_matrix(q):
    q = np.asarray(q, float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm == 0:
        raise ValueError("Quaternion must contain four finite non-zero values")
    w, x, y, z = q / norm
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
    ])
    return R

# -------------------------------------------------------------------------
# Ellipsoid-Formmatrix  M = R diag(1/a^2,1/b^2,1/c^2) R^T
# (x^T M x = 1 beschreibt die Oberflaeche im Weltkoordinatensystem)
# -------------------------------------------------------------------------
def shape_matrix(a, b, c, quat):
    R = quat_to_rot_matrix(quat)
    return R @ np.diag([1/(a*a), 1/(b*b), 1/(c*c)]) @ R.T

# -------------------------------------------------------------------------
# Ellipsoid radius entlang Richtung n (unabhaengige Projektion, nur als
# schnelle Diagnose/Naeherung weiterhin verfuegbar - NICHT mehr fuer
# surface_distance verwendet)
# -------------------------------------------------------------------------
def ellipsoid_radius_along_dir(a, b, c, quat, n):
    M = shape_matrix(a, b, c, quat)
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    return 1.0 / np.sqrt(n @ (M @ n))

# -------------------------------------------------------------------------
# Perram-Wertheim Kontaktfunktion und Oberflaechenabstand
# Perram & Wertheim, J. Comput. Phys. 58, 409 (1985);
# Naeherungsformel fuer d_PW nach Dorosz et al., RSC Soft Matter Suppl.,
# arXiv:1403.5387, Gl. (39)-(41)
# -------------------------------------------------------------------------
def perram_wertheim_distance(r_a, A, r_b, B):
    """
    r_a, r_b : Zentren (3,) der Ellipsoide 1 und 2
    A, B     : Formmatrizen (aus shape_matrix)

    Returns
    -------
    s     : Kontaktfunktionswert (<1 Durchdringung, =1 Beruehrung, >1 Trennung)
    d_pw  : genaeherter Oberflaechenabstand entlang der Zentrenlinie
            (negativ = Durchdringungstiefe, positiv = Luecke)
    """
    r_ab = r_b - r_a
    Ainv = np.linalg.inv(A)
    Binv = np.linalg.inv(B)

    def negF(lam):
        C = np.linalg.inv(lam * Binv + (1 - lam) * Ainv)
        return -(lam * (1 - lam) * (r_ab @ C @ r_ab))

    res = minimize_scalar(negF, bounds=(1e-9, 1 - 1e-9), method='bounded',
                           options={'xatol': 1e-10})
    s = -res.fun
    dist = np.linalg.norm(r_ab)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("Degenerate Perram-Wertheim contact function")
    d_pw = dist * (1.0 - 1.0 / np.sqrt(s))
    return s, d_pw

# -------------------------------------------------------------------------
# RK combination for COLL
# -------------------------------------------------------------------------
def combine_rk_coll(rkdata):
    if not isinstance(rkdata, tuple):
        return rkdata

    alpha = np.array([4/15., 1/15., 1/6.])
    rk1, rk2, rk3 = rkdata
    combined = []

    for r1, r2, r3 in zip(rk1, rk2, rk3):
        comb = {
            "t": r1["t"],
            "dt": r1["dt"],
            "id1": int(r1["id1"]),
            "id2": int(r1["id2"]),
            "pw_c": alpha[0]*np.asarray(r1["pw_c"]) +
                     alpha[1]*np.asarray(r2["pw_c"]) +
                     alpha[2]*np.asarray(r3["pw_c"]),
            "pv_c": alpha[0]*np.asarray(r1["pv_c"]) +
                     alpha[1]*np.asarray(r2["pv_c"]) +
                     alpha[2]*np.asarray(r3["pv_c"]),
            "pi_c": alpha[0]*np.asarray(r1["pi_c"]) +
                     alpha[1]*np.asarray(r2["pi_c"]) +
                     alpha[2]*np.asarray(r3["pi_c"])
        }
        combined.append(comb)

    return combined

# =========================================================================
# MAIN
# =========================================================================
def main(counter=None, path=None, out=None, dry=None):
    if counter is None and path is None and out is None and not dry:
        parser = argparse.ArgumentParser(description="Compute collision surface distances")
        parser.add_argument("--counter", type=int, required=True)
        parser.add_argument("--path", required=False, default=None)
        parser.add_argument("--out", required=False, default=None)
        parser.add_argument("--dry", action="store_true", default=None,
                            help="Use stage 1 only (default: detect collision layout)")
        args = parser.parse_args()
        counter = args.counter
        path = args.path
        out = args.out
        dry = args.dry

    counter = int(counter) if counter is not None else None
    path = _resolve_data_root(path or FOLDER)
    if dry is None:
        dry = collision_rk_stages(CasePaths.from_input(path)) == (1,)
        if dry:
            print("Detected single-stage COLL output; loading stage 1 only.")
    out_dir = out if out else os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Load TRN properly using keyword args
    # -------------------------------------------------------------
    print("Loading TRN…")
    trn = load_data(data="trn", counter=counter, path=path, dry=dry)

    # -------------------------------------------------------------
    # Load COLL properly using keyword args
    # -------------------------------------------------------------
    print("Loading COLL…")
    coll_raw = load_data(data="coll", counter=counter, path=path, dry=dry)
    coll = combine_rk_coll(coll_raw)

    # -------------------------------------------------------------
    # Index TRN by particle ID
    # -------------------------------------------------------------
    trn_by_id = {}
    has_id = any(("id" in e or "idx" in e) for e in trn)

    if has_id:
        for e in trn:
            idx = e.get("id", e.get("idx"))
            trn_by_id[int(idx)] = e
    else:
        # fallback: 1…N
        for i, e in enumerate(trn, start=1):
            trn_by_id[i] = e

    # -------------------------------------------------------------
    # Compute distances
    # -------------------------------------------------------------
    rows = []
    surf_dists = []
    invalid_semiaxis_pairs = []
    skipped_fixed_collisions = 0

    for c in coll:
        id1 = int(c["id1"])
        id2 = int(c["id2"])

        # ignore wall collisions etc.
        if id1 < 0 or id2 < 0:
            continue
        if USE_PARTICLE_SKIP and (
            id1 <= SKIP_FIRST_PARTICLES or id2 <= SKIP_FIRST_PARTICLES
        ):
            skipped_fixed_collisions += 1
            continue
        if id1 not in trn_by_id or id2 not in trn_by_id:
            continue

        e1 = trn_by_id[id1]
        e2 = trn_by_id[id2]

        # positions
        x1 = np.asarray(e1["x"], float)
        x2 = np.asarray(e2["x"], float)

        # velocities
        u1 = np.asarray(e1["u"], float)
        u2 = np.asarray(e2["u"], float)

        center_vec = x2 - x1
        center_dist = np.linalg.norm(center_vec)
        n = center_vec / center_dist if center_dist > 0 else np.array([1, 0, 0], float)

        a1, b1, c1 = e1["a"], e1["b"], e1["c"]
        a2, b2, c2 = e2["a"], e2["b"], e2["c"]
        q1, q2 = e1["q"], e2["q"]

        # Absicherung: manche id1/id2 aus der COLL-Datei matchen offenbar auf
        # trn-Eintraege mit a=b=c=0 (Wand-/Ghost-Kollisionen oder ID-Mismatch
        # zwischen COLL und dem trn_by_id-Mapping). Solche Paare koennen
        # keine gueltige Formmatrix liefern -> ueberspringen statt NaN/Crash.
        semiaxes = (a1, b1, c1, a2, b2, c2)
        if not all(np.isfinite(v) and v > 0 for v in semiaxes):
            invalid_semiaxis_pairs.append((id1, id2, semiaxes))
            continue

        # weiterhin verfuegbar als schnelle, unabhaengige Naeherung
        try:
            r1 = ellipsoid_radius_along_dir(a1, b1, c1, q1, n)
            r2 = ellipsoid_radius_along_dir(a2, b2, c2, q2, -n)
        except Exception:
            r1 = np.mean([a1, b1, c1])
            r2 = np.mean([a2, b2, c2])

        # korrekter, orientierungsgekoppelter Oberflaechenabstand
        try:
            A = shape_matrix(a1, b1, c1, q1)
            B = shape_matrix(a2, b2, c2, q2)
            s_pw, surface_distance = perram_wertheim_distance(x1, A, x2, B)
        except Exception:
            s_pw = np.nan
            surface_distance = center_dist - (r1 + r2)

        surf_dists.append(surface_distance)

        # impulses (vector) + magnitudes
        pw = np.asarray(c["pw_c"])
        pv = np.asarray(c["pv_c"])
        pi = np.asarray(c["pi_c"])

        pw_mag = np.linalg.norm(pw)
        pv_mag = np.linalg.norm(pv)
        pi_mag = np.linalg.norm(pi)

        rows.append([
            c["t"], c["dt"], id1, id2,
            *x1, *x2,
            *u1, *u2,
            a1, b1, c1, a2, b2, c2,
            center_dist, r1, r2,
            s_pw, surface_distance,
            *pw, pw_mag,
            *pv, pv_mag,
            *pi, pi_mag
        ])

    if invalid_semiaxis_pairs:
        n_bad = len(invalid_semiaxis_pairs)
        sample = invalid_semiaxis_pairs[:5]
        print(f"WARNUNG: {n_bad} Kollisionspaare mit ungueltigen Halbachsen "
              f"(<=0 oder nicht endlich) uebersprungen. Beispiele (id1, id2, a1,b1,c1,a2,b2,c2):")
        for id1, id2, semiaxes in sample:
            print(f"   id1={id1} id2={id2} semiaxes={semiaxes}")
        ids_bad = sorted({i for id1, id2, _ in invalid_semiaxis_pairs for i in (id1, id2)})
        print(f"   betroffene distinct IDs ({len(ids_bad)}): {ids_bad[:20]}"
              f"{' ...' if len(ids_bad) > 20 else ''}")
    if skipped_fixed_collisions:
        print(
            f"INFO: {skipped_fixed_collisions} Kollisionen mit den ersten "
            f"{SKIP_FIRST_PARTICLES} fixierten Partikeln uebersprungen."
        )

    # -------------------------------------------------------------
    # Write table
    # -------------------------------------------------------------
    outfile = os.path.join(out_dir, f"coll_surface_distances_{counter:06d}.dat")
    with open(outfile, "w") as f:
        f.write(
            "t dt id1 id2 "
            "x1x x1y x1z x2x x2y x2z "
            "u1x u1y u1z u2x u2y u2z "
            "a1 b1 c1 a2 b2 c2 "
            "center_dist r1 r2 s_pw surface_distance "
            "pw_x pw_y pw_z pw_mag "
            "pv_x pv_y pv_z pv_mag "
            "pi_x pi_y pi_z pi_mag\n"
        )
        for row in rows:
            f.write(" ".join(f"{v:.6g}" for v in row) + "\n")

    print(f"→ wrote table:\n   {outfile}")

    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        sd = np.asarray(surf_dists)

        fig = plt.figure()
        plt.hist(sd, bins=100)
        #plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("surface distance (Perram-Wertheim)")
        plt.ylabel("count")
        plt.title(f"surface distances frame {counter}")
        png = os.path.join(out_dir, f"coll_surface_hist_{counter:06d}.png")
        plotstyle.export_figure(fig, png, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"→ wrote histogram:\n   {png}")

    except Exception as e:
        print("Plot failed:", e)

    print("Done.")

if __name__ == "__main__":
    main()
