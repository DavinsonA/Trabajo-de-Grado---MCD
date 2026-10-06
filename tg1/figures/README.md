# Figuras de TG I

Correspondencia entre las figuras del documento de TG I y los archivos del repositorio.

| Figura del documento | Archivo | Generada por |
|---|---|---|
| Fig. 1. Variables del área de estudio | No incluida | Fuera de este pipeline |
| Fig. 2. Series anuales de temperatura | `../results/05_01_eda_ajustado/fig01_annual_timeseries_tas_adjusted.png` | `05_01_eda_ajustado.py` |
| Fig. 3. Series anuales de humedad relativa | `../results/05_01_eda_ajustado/fig02_annual_timeseries_hurs_adjusted.png` | `05_01_eda_ajustado.py` |
| Fig. 4. Correlación tas–hurs | `../results/05_01_eda_ajustado/fig03_correlation_tas_hurs_adjusted.png` | `05_01_eda_ajustado.py`: Pearson sobre valores mensuales sin desestacionalizar, 3 modelos (ACCESS-CM2, CESM2, CanESM5) |
| Fig. 5. Pendiente de Sen de tas | `sen_slope_tas.png` | `tg1_static_figures.ipynb` (origen inferido) |
| Fig. 6. Pendiente de Sen de hurs | `sen_slope_hurs.png` | `tg1_static_figures.ipynb` (origen inferido) |
| Fig. 7. Clasificación bivariada calentamiento–desecamiento | `fig6_bivariate_map.png` | `tg1_static_figures.ipynb` |
| Fig. 8. Desplazamiento conjunto T–H | `fig7_th_arrows.png` | `tg1_static_figures.ipynb` (en el documento, sin el prefijo "Figure 7 —" en el título) |
| Fig. 9. Matriz de acoplamiento T–H | `fig8_coupling_matrix.png` | `tg1_static_figures.ipynb` (en el documento, sin el prefijo "Figure 8 —" en el título) |
| Fig. 10. Tablero interactivo | Captura de pantalla | `tg1_dashboard.ipynb` |

## Otros archivos

- `headline_defense_v2.png`: figura de cierre de la sustentación.
- `fig2_temperature_divergence.png`, `fig3_tas_heatmap.png`, `fig4_spatial_tas.png` y `fig5_hurs_heatmap.png`: generadas por `tg1_static_figures.ipynb`; no se usaron en el documento.
- `fig1_study_area.png`: no se usó en el documento. Sus recuadros no corresponden a la definición de subregiones del pipeline, que son bandas de latitud de ancho completo (ver `docs/data_layout.md`).
