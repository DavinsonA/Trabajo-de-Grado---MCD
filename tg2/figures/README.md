# Figuras de TG II

Correspondencia entre las figuras del documento de TG II y los archivos del repositorio. Cada panel lleva una letra (A, B, …) en su título, que el pie de figura describe. Los scripts guardan las figuras a 300 dpi.

| Figura del documento | Archivo | Generada por |
|---|---|---|
| Fig. 1. Variables del área de estudio en un día de ejemplo (A–E) | `fig1_study_variables.png` | Fuera de este pipeline: es la Fig. 1 de TG I, con las letras A–E agregadas sobre la imagen y sin su título general |
| Fig. 2. Concentraciones de gases por escenario: (A) CO₂-eq, (B) CO₂ | `forcing_trajectories.png` | `02_forcing_meinshausen.py` |
| Fig. 3. Respuesta forzada y umbral de emergencia, SSP1-2.6 y SSP5-8.5: (A–D) temperatura, (E–H) humedad relativa | `fan_charts_toe.png` | `06_toe_divergence.py` |
| Fig. 4. Puntos de cambio de pendiente y varianza bajo SSP5-8.5, (A–D) por subregión | `pelt_changepoints.png` | `07_pelt_changepoints.py` |
| Fig. 5. Covarianza T–H normalizada bajo SSP1-2.6 y SSP5-8.5, (A–D) por subregión | `th_coupling.png` | `07_pelt_changepoints.py` |
| Fig. 6. Fracción de la varianza por fuente de incertidumbre, 2015–2100: (A–D) temperatura, (E–H) humedad relativa | `uncertainty_partition.png` | `08_uncertainty_partition.py` (8 modelos, medias decadales) |
| Fig. 7. Mejora del CRPS frente a la línea base y cobertura del intervalo del 90 %, por fold: (A, B) temperatura, (C, D) humedad relativa | `oe4_skill_coverage.png` | `10_oe4_figures.py`, a partir de `oe4_metrics.csv` de `09_emulators.py` |
| Fig. 8. Importancia SHAP por grupo de variables: (A) temperatura, (B) humedad relativa | `oe4_shap_importance.png` | `09_emulators.py` |

## Otros archivos

- `toe_maps_ssp585.png`: mapas del año de emergencia por celda bajo SSP5-8.5, (A) temperatura y (B) humedad relativa, generados por `12_toe_maps.py`. El texto del documento menciona estos mapas, pero la figura no está incluida.
