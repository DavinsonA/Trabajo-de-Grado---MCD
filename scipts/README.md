# NEX-GDDP-CMIP6 — Pipeline de Extracción para Sudamérica

Esta carpeta contiene los scripts utilizados para preparar el entorno, extraer datos climáticos NEX-GDDP CMIP6 recortados para Sudamérica, y verificar la extracción.

Los archivos principales son:
- `01_setup_environment.sh`: crea el entorno virtual y las dependencias necesarias en la máquina remota.
- `02_extract_nexgddp.py`: extrae los datos de los modelos/climatologías y los recorta a la región sudamericana.
- `03_verify_extraction.py`: verifica que la extracción haya generado los archivos esperados y que no falte información.
- `04_dataset_inventory.py`, `05_eda_climate.py`: scripts posteriores para inventario del conjunto de datos y análisis exploratorio.


## Notas importantes
- CESM2 usa variante `r4i1p1f1` (los demás usan `r1i1p1f1`)
- El script es resumible: si se interrumpe, al reiniciarlo salta archivos ya descargados
- Logs se guardan en `/media/volume/jay2vol/nexgddp/logs/`
- Los archivos temporales globales (~250 MB cada uno) se borran automáticamente después del recorte
