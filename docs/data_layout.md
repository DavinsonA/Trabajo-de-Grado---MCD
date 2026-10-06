# Organización de los datos en la máquina de trabajo

Los datos no se versionan en git. Viven en el volumen de la máquina virtual:

```
/media/volume/jay2vol/nexgddp/
├── raw/          datos descargados y recortados (6,544 archivos NetCDF, ~152 GB)
├── processed/    tablas generadas por 06_preprocessing.py
├── outputs/      resultados de los scripts 04–10
├── logs/         registros de la extracción
├── tmp/          temporales de la extracción
└── venv/         entorno de Python
```

## raw/

Ruta de cada archivo: `raw/<MODEL>/<SCENARIO>/<VARIABLE>/<variable>_<MODEL>_<scenario>_<year>_SA.nc`

- Un archivo por año: 65 en historical (1950–2014) y 86 por cada SSP (2015–2100), es decir, 409 por modelo y variable.
- Grilla de 276 × 192 celdas de 0.25° (56°S–13°N, 82°W–34°W). Cerca del 51 % de las celdas es océano y viene como NaN.
- Variante: `r1i1p1f1` en todos los modelos excepto CESM2 (`r4i1p1f1`).
- Origen: `s3://nex-gddp-cmip6/NEX-GDDP-CMIP6/<MODEL>/<SCENARIO>/<VARIANT>/<VARIABLE>/`.

## Subregiones

Bandas de latitud que abarcan todo el ancho del recuadro (82°W–34°W):

| Subregión | Latitud |
|---|---|
| Northern Tropics | 0° a 13°N |
| Central Amazonia | 15°S a 0° |
| Subtropical Andes | 30°S a 15°S |
| Southern Cone | 56°S a 30°S |

La media de cada subregión es el promedio simple de sus celdas de tierra, sin ponderar por latitud. El agregado "continental" es la media simple de las cuatro subregiones.

## processed/

| Archivo | Contenido |
|---|---|
| `monthly_subregion_means.csv` | Media mensual por modelo, escenario, variable y subregión (314,112 filas) |
| `climatology_1960_2014.csv` | Climatología mensual de referencia 1960–2014, por modelo |
| `monthly_subregion_anomalies.csv` | Anomalías mensuales respecto a esa climatología |
| `annual_subregion_anomalies.csv` | Anomalías anuales por subregión |
| `monthly_continental_anomalies.csv`, `annual_continental_anomalies.csv` | Anomalías del agregado continental |
| `ensemble_annual_stats.csv` | Percentiles, media y desviación entre modelos, por año |
| `spatial_decadal/*.nc` | Mapas de anomalías por década (media de los 8 modelos; décadas futuras bajo SSP5-8.5) |

`tas` se guarda en °C (convertida desde K) y `hurs` en %.
