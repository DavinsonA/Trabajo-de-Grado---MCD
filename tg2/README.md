# Trabajo de Grado II

**Scenario Divergence and Projection Uncertainty of Temperature and Humidity in South America (1950–2100): A Probabilistic Machine Learning Approach**

Objetivos específicos y dónde se resuelve cada uno:

| Objetivo | Scripts | Figuras del documento |
|---|---|---|
| OE1. Ventana de divergencia entre escenarios y tiempo de emergencia por subregión | `06_toe_divergence.py`, `11_cell_annual_means.py`, `12_toe_maps.py` | Fig. 3 |
| OE2. Puntos de cambio (PELT) en las tendencias y en el acoplamiento temperatura–humedad | `07_pelt_changepoints.py` | Figs. 4 y 5 |
| OE3. Partición de la incertidumbre: escenario, inter-modelo y variabilidad interna | `08_uncertainty_partition.py` | Fig. 6 |
| OE4. Emuladores probabilísticos de machine learning, evaluados con CRPS e interpretados con SHAP | `02_forcing_meinshausen.py`, `04_emulator_table.py`, `05_cost_benchmark.py`, `09_emulators.py`, `10_oe4_figures.py` | Figs. 2, 7 y 8 |

Los scripts `01` y `03` construyen la base común de los cuatro objetivos. El orden de ejecución y las dependencias están en el [README principal](../README.md#reproducir-tg-ii); la correspondencia entre figuras y archivos, en [`figures/README.md`](figures/README.md).

## Decisiones de diseño

**Datos base**

- Tabla base: columna `mean_weighted_qc` de `processed_tg2/monthly_region_means.parquet`. Es la media mensual de las celdas de tierra ponderada por cos(latitud), con la `hurs` diaria recortada a [0, 100]. El recorte afectó al 0.035 % de los días-celda de tierra de `hurs` y cambia las anomalías regionales en 0.0008 % o menos.
- Regiones: las cuatro bandas de latitud de TG I y `All land`, el agregado continental ponderado por área. En TG I el agregado continental era el promedio simple de las cuatro bandas.
- La pasada de `01` reproduce la tabla mensual de TG I con su método (media simple): diferencia máxima de 0.00016 en 314,112 filas.

**Anomalías y respuesta forzada**

- Anomalías respecto a la climatología 1960–2014 de cada modelo, variable, región y mes calendario. La anomalía anual es la media de las 12 anomalías mensuales del año.
- El histórico se guarda una vez por modelo y se concatena con cada SSP para formar las series 1950–2100.
- σ_ref: desviación estándar de las anomalías anuales 1960–2014 después de quitar una tendencia lineal, por modelo, variable y región.
- Respuesta forzada: un polinomio de grado 4 por serie modelo–SSP–variable–región, 1950–2100 (Hawkins y Sutton, 2009). Lo reutilizan los objetivos 1 a 3.

**OE1**

- Tiempo de emergencia: primer año desde el cual la relación señal/ruido (respuesta forzada / σ_ref), orientada según el signo de la señal en 2100, se mantiene ≥ 2 hasta 2100.
- Divergencia: primer año, desde 2015, en que la diferencia entre las respuestas forzadas de dos escenarios de un mismo modelo supera √2·σ_ref y se mantiene así hasta 2100. Se reporta también con el umbral estricto 2·√2·σ_ref.
- Mapas: el mismo método por celda de tierra. El resumen del ensamble es la mediana de los 8 modelos, contando la no emergencia como posterior a 2100. Una celda emerge solo si emergen al menos 5 de los 8 modelos.

**OE2**

- PELT (librería `ruptures`) sobre los datos, no sobre la curva suavizada:
  - pendiente: regresión lineal por tramos sobre las anomalías anuales;
  - varianza: segmentación gaussiana de los residuos alrededor de la respuesta forzada;
  - acoplamiento: cambios de media en la media anual del producto de los residuos mensuales de `tas` y `hurs`, estandarizados con su desviación de 1960–2014 (covarianza T–H normalizada).
- La penalización se calibra, por tipo, variable y región, con series AR(1) sin cambios: la menor de la grilla con tasa de falsos positivos ≤ 5 %.
- Un punto de cambio se reporta solo si al menos 5 de los 8 modelos tienen uno en el mismo conglomerado (puntos separados por 5 años o menos). Los anteriores a 2015 se agrupan por modelo entre escenarios y se reportan como `historical`.
- Como medida descriptiva complementaria, `03` calcula la correlación móvil de Spearman sin tendencia (ventana de 30 años). Su autocorrelación de un paso es 0.96–0.97, por lo que no se usa como entrada de PELT.

**OE3**

- Partición de Hawkins y Sutton (2009) sobre las respuestas forzadas re-referidas a 1995–2014, evaluada en 2015–2100 y en los horizontes del AR6 (2021–2040, 2041–2060 y 2081–2100).
- Resultado principal en medias decadales; la escala anual queda como sensibilidad. Se calcula con los 8 modelos y con el subconjunto restringido por TCR.

**OE4**

- Forzamiento: concentraciones de Meinshausen et al. (2020), forzamiento radiativo de gases según Myhre et al. (1998) y CO₂-eq. El emulador recibe el forzamiento en W m⁻².
- Variables de entrada: forzamiento actual y su media de 20 años, TCR, mes (seno y coseno) y subregión (one-hot). El conjunto sin la media de 20 años (`base`) es la ablación; el conjunto con forzamiento acumulado está definido en `emulator_folds.json` pero no se evaluó.
- Sensibilidad de cada modelo: TCR de Meehl et al. (2020), Tabla 2. El modelo entra al emulador por su TCR, no como identificador; el ECS queda en la tabla, pero no se usa como variable de entrada.
- Subconjunto restringido (TCR 1.4–2.2 °C): ACCESS-CM2, CESM2, GFDL-ESM4, MIROC6 y MPI-ESM1-2-HR.
- Validación: 4 folds dejando fuera un escenario (prueba: 2015–2100 del escenario retenido) y 8 folds dejando fuera un modelo (prueba: todas sus filas). El histórico compartido queda siempre en entrenamiento en los folds por escenario.
- Emuladores:
  - regresión lineal como línea base;
  - XGBoost con pérdida cuantil (19 cuantiles, 5 % a 95 %);
  - ensamble de 5 redes heterocedásticas (2 × 64 neuronas, pérdida gaussiana).
- Métricas: CRPS aproximado sobre la misma grilla de 19 cuantiles para los tres emuladores; RMSE mensual y de medias anuales; cobertura y ancho del intervalo del 90 %. SHAP con TreeSHAP nativo de XGBoost sobre el modelo de mediana.
- Hiperparámetros fijos, sin búsqueda.

## Contenido de `data/`

| Archivo | Contenido | Generado por |
|---|---|---|
| `forcing_ssp.parquet` | 409 filas (historical 1950–2014 y cada SSP 2015–2100): `co2eq_ppm`, `co2_ppm`, `ghg_rf_wm2`, `ghg_rf_ma20`, `ghg_rf_cum` | `02_forcing_meinshausen.py` |
| `gcm_sensitivity.csv` | ECS, TCR, fuente del TCR y pertenencia al subconjunto restringido, por modelo | Manual, desde Meehl et al. (2020), Tabla 2 |
| `emulator_folds.json` | Variables objetivo, conjuntos de variables de entrada, reglas y tamaños de los 12 folds | `04_emulator_table.py` |

## Limitaciones conocidas

- El forzamiento incluye solo gases de efecto invernadero, no aerosoles. En 2100 supera la cifra nominal de cada SSP en 0.5–0.9 W m⁻².
- EC-Earth3 no tiene TCR en la Tabla 2 de Meehl et al. (2020); se usa el de EC-Earth3-Veg (2.6 °C) como supuesto declarado.
- Hay una sola realización por modelo (variante `r1i1p1f1`; `r4i1p1f1` en CESM2), así que la variabilidad interna se estima de los residuos de esa realización.
- Esconder SSP5-8.5 obliga a extrapolar el forzamiento (9.18 W m⁻² frente a un máximo de 7.87 W m⁻² en entrenamiento).
- El CRPS sobre 19 cuantiles omite las colas por debajo del 5 % y por encima del 95 %. Afecta por igual a los tres emuladores.
- Los modelos con igual TCR (GFDL-ESM4 y MIROC6) no se distinguen en el emulador.
- La serie de acoplamiento de `07` es una covarianza normalizada con la desviación de 1960–2014, no una correlación: si la variabilidad aumenta en el siglo XXI, la serie crece aunque la correlación no cambie.
