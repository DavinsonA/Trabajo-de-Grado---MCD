# Trabajo de Grado — Maestría en Ciencia de Datos (Universidad Icesi)

Tendencias, divergencia entre escenarios e incertidumbre de la temperatura del aire (`tas`) y la humedad relativa (`hurs`) en Sudamérica, 1950–2100, con el dataset NASA NEX-GDDP-CMIP6.

| Fase | Título | Estado |
|---|---|---|
| TG I | Temperature and Humidity Trends in South America (1950–2100): Spatiotemporal Analysis Using the NEX-GDDP-CMIP6 Dataset | Entregado. Versión congelada en el tag `tg1-final` |
| TG II | Scenario Divergence and Projection Uncertainty of Temperature and Humidity in South America (1950–2100): A Probabilistic Machine Learning Approach | Pipeline completo (scripts 01–12); documento en revisión final, noviembre de 2026 |

- **Autores:** Davinson Alexander Arteaga Bermudez y Gian Alepsi Mendoza Oviedo
- **Director:** PhD. Jack Marquez

## Datos

- Fuente: NEX-GDDP-CMIP6 (Thrasher et al., 2022, doi:10.1038/s41597-022-01393-4), bucket público `s3://nex-gddp-cmip6`, licencia CC BY-SA 4.0.
- Recorte: 56°S–13°N, 82°W–34°W, resolución 0.25°.
- 8 GCMs: ACCESS-CM2, CanESM5, CESM2, EC-Earth3, GFDL-ESM4, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR.
- Escenarios: historical (1950–2014); SSP1-2.6, SSP2-4.5, SSP3-7.0 y SSP5-8.5 (2015–2100).
- Variables: `tas` y `hurs`.
- TG II usa además las concentraciones de gases de efecto invernadero de Meinshausen et al. (2020) y la sensibilidad climática de cada modelo de Meehl et al. (2020).

Los datos (~152 GB) no están en este repositorio. Su organización en la máquina de trabajo se describe en [`docs/data_layout.md`](docs/data_layout.md).

## Estructura

```
.
├── docs/            organización de los datos y referencias
├── tg1/
│   ├── pipeline/    scripts 01–10, en orden de ejecución
│   ├── notebooks/   tablero interactivo, notebook de figuras y analysis_lib.py
│   ├── results/     tablas y figuras generadas por los scripts
│   └── figures/     figuras del documento de TG I (ver su README)
└── tg2/
    ├── pipeline/    scripts 01–12 (ver tabla abajo)
    ├── data/        insumos pequeños versionados: forzamiento, sensibilidad de los modelos, folds
    └── figures/     figuras del documento de TG II (ver su README)
```

Entorno (Python 3.12.3), común a ambas fases:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` instala PyTorch en su versión solo para CPU, desde el índice indicado en su primera línea.

## Reproducir TG I

Scripts de `tg1/pipeline/`, en orden:

| Script | Qué hace | Copia de la salida en el repo |
|---|---|---|
| `01_setup_environment.sh` | Crea el entorno y las carpetas en la máquina | — |
| `02_extract_nexgddp.py` | Descarga desde S3 y recorta a Sudamérica (6,544 archivos anuales) | — |
| `03_verify_extraction.py` | Verifica la descarga | — |
| `04_dataset_inventory.py` | Inventario y control de calidad (Apéndice A de TG I) | `tg1/results/04_inventory` |
| `05_eda_climate.py` | Análisis exploratorio | `tg1/results/05_eda` |
| `05_01_eda_ajustado.py` | Figuras 2–4 de TG I | `tg1/results/05_01_eda_ajustado` |
| `06_preprocessing.py` | Medias mensuales por subregión, climatología 1960–2014, anomalías y estadísticos de ensamble (varias horas) | — |
| `07_trend_analysis.py` | Mann-Kendall (corrección de Hamed-Rao) y pendiente de Sen | `tg1/results/07_trends` |
| `08_uncertainty_decomposition.py` | Descomposición de incertidumbre de Hawkins y Sutton | `tg1/results/08_hawkins_sutton` |
| `09_temp_humidity_correlation.py` | Correlación tas–hurs sobre anomalías mensuales | `tg1/results/09_correlation` |
| `10_spatial_trends.py` | Tendencias por celda de la grilla (NetCDF, no versionados) | — |

Los notebooks de `tg1/notebooks/` deben ejecutarse desde esa carpeta, porque importan `analysis_lib.py` desde ella.

Scripts y notebooks usan la ruta absoluta `/media/volume/jay2vol/nexgddp` (variable `BASE_DIR`), donde escriben sus salidas. Para ejecutarlos en otra máquina hay que ajustarla. `tg1/results/` contiene la copia de las salidas usadas en TG I.

## Reproducir TG II

TG II parte de los mismos archivos diarios de `raw/` (descargados con los scripts 02–03 de TG I) y no usa las tablas de TG I. Los scripts leen la raíz de datos de la variable de entorno `NEXGDDP_ROOT` (por defecto `/media/volume/jay2vol/nexgddp`) y escriben sus tablas en `processed_tg2/`; las figuras y los insumos pequeños van a `tg2/figures/` y `tg2/data/` dentro del repo. Los scripts marcados como largos son reanudables y conviene correrlos dentro de `tmux`.

| Script | Qué hace | Objetivo | Depende de |
|---|---|---|---|
| `01_subregional_monthly_means.py` | Una pasada sobre los 6,544 archivos diarios: media mensual por modelo, escenario, variable y región, simple, ponderada por cos(lat) y ponderada con `hurs` recortada a [0, 100] (largo, reanudable) | Base | `raw/` |
| `02_forcing_meinshausen.py` | Concentraciones, forzamiento de gases y CO₂-eq por escenario; Fig. 2 | OE4 | `external/meinshausen2020/` |
| `03_anomalies_forced_response.py` | Anomalías mensuales y anuales, σ_ref, respuesta forzada (polinomio de grado 4) y correlación móvil tas–hurs | Base | 01 |
| `04_emulator_table.py` | Tabla del emulador (157,056 filas por variable) y 12 folds de validación con control de fugas | OE4 | 02, 03 |
| `05_cost_benchmark.py` | Prueba de costo computacional en un fold (opcional) | OE4 | 04 |
| `06_toe_divergence.py` | Tiempo de emergencia y año de divergencia entre escenarios por subregión; Fig. 3 | OE1 | 03 |
| `07_pelt_changepoints.py` | Puntos de cambio con PELT en pendiente, varianza y acoplamiento T–H, con penalización calibrada y consenso entre modelos; Figs. 4 y 5 | OE2 | 03 |
| `08_uncertainty_partition.py` | Partición de la incertidumbre (Hawkins y Sutton): interna, modelo y escenario; Fig. 6 | OE3 | 03 |
| `09_emulators.py` | Emuladores lineal, XGBoost cuantil y ensamble de redes en los 12 folds; CRPS, cobertura y SHAP; Fig. 8 (~4 h, reanudable) | OE4 | 04 |
| `10_oe4_figures.py` | Mejora frente al lineal y cobertura por fold; Fig. 7 | OE4 | 09 |
| `11_cell_annual_means.py` | Medias anuales por celda de tierra (historical, SSP2-4.5, SSP5-8.5) (largo, reanudable) | OE1 | `raw/` |
| `12_toe_maps.py` | Tiempo de emergencia por celda y mapas | OE1 | 11 |

El suplemento de Meinshausen et al. (2020) se descarga una vez:

```bash
mkdir -p /media/volume/jay2vol/nexgddp/external/meinshausen2020 && cd $_
curl -L -o supplement.zip https://gmd.copernicus.org/articles/13/3571/2020/gmd-13-3571-2020-supplement.zip
unzip -o supplement.zip
```

Las decisiones de diseño de TG II y el contenido de `tg2/data/` están en [`tg2/README.md`](tg2/README.md). Las referencias están en [`docs/references.md`](docs/references.md).
