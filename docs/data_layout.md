# Organización de los datos en la máquina de trabajo

Los datos no se versionan en git. Viven en el volumen de la máquina virtual:

```
/media/volume/jay2vol/nexgddp/
├── raw/            datos descargados y recortados (6,544 archivos NetCDF, ~152 GB)
├── processed/      tablas de TG I, generadas por tg1/pipeline/06_preprocessing.py
├── processed_tg2/  tablas de TG II, generadas por tg2/pipeline/
├── external/       datos auxiliares de terceros (Meinshausen et al., 2020)
├── outputs/        resultados de los scripts 04–10 de TG I
├── logs/           registros de la extracción
├── tmp/            temporales de la extracción
└── venv/           entorno de Python
```

## raw/

Ruta de cada archivo: `raw/<MODEL>/<SCENARIO>/<VARIABLE>/<variable>_<MODEL>_<scenario>_<year>_SA.nc`

- Un archivo por año: 65 en historical (1950–2014) y 86 por cada SSP (2015–2100), es decir, 409 por modelo y variable.
- Grilla de 276 × 192 celdas de 0.25° (56°S–13°N, 82°W–34°W). 25,829 celdas (48.74 %) son tierra; el resto es océano y viene como NaN, con la misma máscara en todos los archivos.
- Variante: `r1i1p1f1` en todos los modelos excepto CESM2 (`r4i1p1f1`).
- Origen: `s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/<MODEL>/<SCENARIO>/<VARIANT>/<VARIABLE>/`.

## Subregiones

Bandas de latitud que abarcan todo el ancho del recuadro (82°W–34°W):

| Subregión | Latitud | Celdas de tierra |
|---|---|---|
| Northern Tropics | 0° a 13°N | 4,264 |
| Central Amazonia | 15°S a 0° | 9,859 |
| Subtropical Andes | 30°S a 15°S | 6,677 |
| Southern Cone | 56°S a 30°S | 5,029 |
| All land (solo TG II) | 56°S a 13°N | 25,829 |

Ningún centro de celda cae sobre un límite, así que cada celda pertenece a una sola banda.

- **TG I:** la media de cada subregión es el promedio simple de sus celdas de tierra, y el agregado "continental" es la media simple de las cuatro subregiones.
- **TG II:** las medias se ponderan por cos(latitud), la `hurs` diaria se recorta a [0, 100] y el agregado continental es la región `All land`, ponderada por área.

`tas` se guarda en °C (convertida desde K) y `hurs` en %.

## processed/ (TG I)

| Archivo | Contenido |
|---|---|
| `monthly_subregion_means.csv` | Media mensual por modelo, escenario, variable y subregión (314,112 filas) |
| `climatology_1960_2014.csv` | Climatología mensual de referencia 1960–2014, por modelo |
| `monthly_subregion_anomalies.csv` | Anomalías mensuales respecto a esa climatología |
| `annual_subregion_anomalies.csv` | Anomalías anuales por subregión |
| `monthly_continental_anomalies.csv`, `annual_continental_anomalies.csv` | Anomalías del agregado continental |
| `ensemble_annual_stats.csv` | Percentiles, media y desviación entre modelos, por año |
| `spatial_decadal/*.nc` | Mapas de anomalías por década (media de los 8 modelos; décadas futuras bajo SSP5-8.5) |

## processed_tg2/ (TG II)

| Archivo | Contenido | Script |
|---|---|---|
| `chunks/`, `run_log.txt` | Resultados parciales de la pasada sobre `raw/` (80 parquet, uno por modelo, escenario y variable) y su registro | `01` |
| `monthly_region_means.parquet` | Media mensual por modelo, escenario, variable, región, año y mes (392,640 filas): `mean_unweighted`, `mean_weighted`, `mean_weighted_qc`, conteos de control de calidad y de recorte | `01` |
| `monthly_anomalies.parquet`, `annual_anomalies.parquet` | Anomalías respecto a 1960–2014 sobre `mean_weighted_qc` (392,640 y 32,720 filas) | `03` |
| `sigma_ref.csv` | Variabilidad de referencia por modelo, variable y región | `03` |
| `forced_response.parquet`, `forced_response_summary.csv` | Respuesta forzada (polinomio de grado 4) y residuos por serie 1950–2100; resumen del ajuste | `03` |
| `rolling_spearman.parquet` | Correlación tas–hurs en ventanas de 30 años, cruda y sin tendencia (19,520 ventanas) | `03` |
| `emulator_table.parquet` | Tabla del emulador: 157,056 filas por variable objetivo, 21 columnas | `04` |
| `cost_benchmark.json` | Tiempos y memoria de la prueba de costo | `05` |
| `toe_summary.csv`, `divergence_all_pairs.csv`, `divergence_ensemble_summary.csv`, `divergence_ensemble_summary_strict.csv` | Tiempo de emergencia y años de divergencia por modelo y resumen del ensamble (umbrales √2·σ_ref y 2·√2·σ_ref) | `06` |
| `pelt_changepoints_all.csv`, `pelt_changepoints_ensemble.csv`, `pelt_penalties.csv`, `th_coupling_annual.csv` | Puntos de cambio por modelo y de consenso, penalizaciones calibradas y serie anual de acoplamiento | `07` |
| `uncertainty_partition.csv`, `uncertainty_partition_horizons.csv` | Partición de la incertidumbre por año y por horizonte | `08` |
| `oe4/chunks/`, `oe4/oe4_metrics.csv`, `oe4/shap_importance.csv`, `oe4_run.log` | Predicciones y métricas por fold, variable y conjunto de entrada (48 combinaciones); importancia SHAP; registro de la corrida | `09` |
| `cells/` | Medias anuales por celda de tierra (`<MODEL>__<SCENARIO>__<VARIABLE>.npz`, años × celdas) y `grid.npz` con latitud, longitud y máscara | `11` |
| `toe_cells_<scenario>_<variable>.npz`, `toe_cells_summary.csv` | Año de emergencia por celda (mediana del ensamble y número de modelos que emergen) y fracción de superficie emergida en 2040, 2060 y 2100 | `12` |

## external/

| Ruta | Contenido |
|---|---|
| `meinshausen2020/` | Suplemento de Meinshausen et al. (2020): `SUPPLEMENT_DataTables_Meinshausen_6May2020.xlsx`, con concentraciones medias globales y hemisféricas de 43 gases, histórico desde el año 1 y SSP hasta 2500 |
