# Literaturparameter des Skalierungsmoduls

Definitionen im neuen Modul: chi = H0/L0 (Anfangssäulen-Seitenverhältnis),
l = (L_end-L0)/L0, h = H_end/L0. Falls chi in der Arbeit das **Partikelformverhältnis**
bezeichnet, dürfen diese Gesetze nicht darauf angewandt werden. In achsensymmetrischen
Experimenten ist L0 der Anfangsradius. Trockene Experimente sind Referenzen, keine
belegte Vorhersage für eingetauchte DNS.

| Primärquelle / Stelle | Geometrie | l | h |
|---|---|---|---|
| Balmforth & Kerswell 2005, Gl. 4.1–4.2 | breiter Kanal | proportional chi^(0.9 ± 0.1) | proportional chi^0.4, aus H0/H_end proportional chi^0.6 (chi>2) |
| dieselben, Gl. 5.1–5.2 | schmaler Kanal | proportional chi^(0.65 ± 0.05) | proportional chi^0.5 |
| Lajeunesse, Monnier & Homsy 2005, Zusammenfassung S. 103302-14 | rechteckiger Kanal | chi^1 bei chi<3, chi^(2/3) bei chi>3 | chi^1 bei chi<0.7, chi^(1/3) bei chi>0.7 |
| Lube et al. 2004, Gl. 3–4, 8–9 | achsensymmetrisch | 1.24 chi bei chi<1.7; 1.6 chi^(1/2) bei chi>1.7 | chi bei chi<1.7; 0.88 chi^(1/6) bei 1.7<chi<10 |

Balmforth & Kerswell haben materialabhängige Vorfaktoren. Das Modul verlangt daher
für proportionale Gesetze belegte Referenzpunkte (chi_ref, value_ref, source).
Es setzt keine erfundenen Amplituden auf 1 und passt keine Literaturkurve stillschweigend
an DNS-Daten an. Die Potenz 0.4 der Höhe im breiten Kanal folgt durch Umformung:
h = chi/(H0/H_end). Unterschiedliche Regime werden einzeln gezeichnet.

Primärtexte, am 24.09.2026 geprüft:
- [Balmforth & Kerswell: Granular collapse in two dimensions](https://www.researchgate.net/publication/232023995_Granular_collapse_in_two_dimensions), JFM 538, 399–428, DOI 10.1017/S0022112005005537 (Volltext der Originalarbeit).
- [Lajeunesse et al.: Granular slumping on a horizontal surface](https://www.ipgp.fr/~lajeunes/download/2005-pof-slumping.pdf), Physics of Fluids 17, 103302.
- [Lube et al.: Axisymmetric collapses of granular columns](https://www.researchgate.net/publication/231787439_Axisymmetric_collapses_of_granular_columns), JFM 508, 175–199, DOI 10.1017/S0022112004009036 (Volltext der Originalarbeit).
