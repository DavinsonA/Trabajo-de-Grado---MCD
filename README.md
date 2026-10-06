# Trabajo de Grado — Maestría en Ciencia de Datos (Universidad Icesi)

Tendencias e incertidumbre de la temperatura del aire (`tas`) y la humedad relativa (`hurs`) en Sudamérica, 1950–2100, con el dataset NASA NEX-GDDP-CMIP6.

| Fase | Título | Estado |
|---|---|---|
| TG I | Temperature and Humidity Trends in South America (1950–2100): Spatiotemporal Analysis Using the NEX-GDDP-CMIP6 Dataset | Entregado. Versión congelada en el tag `tg1-final` |
| TG II | Scenario Divergence and Projection Uncertainty of Temperature and Humidity in South America (1950–2100): A Probabilistic Machine Learning Approach | En curso |

- **Autores:** Davinson Alexander Arteaga Bermudez y Gian Alepsi Mendoza Oviedo
- **Director:** PhD. Jack Marquez

## Datos

- Fuente: NEX-GDDP-CMIP6 (Thrasher et al., 2022, doi:10.1038/s41597-022-01393-4), bucket público `s3://nex-gddp-cmip6`, licencia CC BY-SA 4.0.
- Recorte: 56°S–13°N, 82°W–34°W, resolución 0.25°.
- 8 GCMs: ACCESS-CM2, CanESM5, CESM2, EC-Earth3, GFDL-ESM4, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR.
- Escenarios: historical (1950–2014); SSP1-2.6, SSP2-4.5, SSP3-7.0 y SSP5-8.5 (2015–2100).
- Variables: `tas` y `hurs`.

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
└── tg2/             Trabajo de Grado II (en curso)
```

## Reproducir TG I

Entorno (Python 3.12.3):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

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

Las lecturas de apoyo están en [`docs/references.md`](docs/references.md).
