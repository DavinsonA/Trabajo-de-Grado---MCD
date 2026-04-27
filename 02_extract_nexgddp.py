#!/usr/bin/env python3
"""
02_extract_nexgddp.py
Pipeline de extracción del dataset NEX-GDDP-CMIP6 desde AWS S3.

Descarga archivos NetCDF anuales globales, recorta a Sudamérica,
y guarda los subconjuntos en el volumen local.

Uso:
    # Activar entorno primero
    source /media/volume/jay2vol/nexgddp/venv/bin/activate

    # Test con 1 modelo, 1 variable, 1 año
    python 02_extract_nexgddp.py --test

    # Extracción completa (ejecutar en tmux)
    python 02_extract_nexgddp.py --full

    # Un modelo específico
    python 02_extract_nexgddp.py --model ACCESS-CM2 --variable tas --scenario historical

    # Solo listar qué se descargaría (dry run)
    python 02_extract_nexgddp.py --full --dry-run
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import xarray as xr
import numpy as np

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR = Path("/media/volume/jay2vol/nexgddp")
RAW_DIR = BASE_DIR / "raw"
TMP_DIR = BASE_DIR / "tmp"
LOG_DIR = BASE_DIR / "logs"

S3_BUCKET = "https://nex-gddp-cmip6.s3.us-west-2.amazonaws.com"
S3_PREFIX = "NEX-GDDP-CMIP6"

# Bounds de Sudamérica
# Latitud: -56°S a 13°N
# Longitud: -82°W a -34°W → en sistema 0-360: 278°E a 326°E
LAT_MIN, LAT_MAX = -56.0, 13.0
LON_MIN_360, LON_MAX_360 = 278.0, 326.0  # Sistema 0-360 (como viene en NetCDF)
LON_MIN_180, LON_MAX_180 = -82.0, -34.0  # Sistema -180 a 180 (por si acaso)

# Modelos seleccionados con sus variantes correctas
MODELS = {
    "ACCESS-CM2":      "r1i1p1f1",
    "CanESM5":         "r1i1p1f1",
    "CESM2":           "r4i1p1f1",   # ¡Atención: variante diferente!
    "EC-Earth3":       "r1i1p1f1",
    "GFDL-ESM4":       "r1i1p1f1",
    "IPSL-CM6A-LR":    "r1i1p1f1",
    "MIROC6":          "r1i1p1f1",
    "MPI-ESM1-2-HR":   "r1i1p1f1",
}

VARIABLES = ["tas", "hurs"]

SCENARIOS = {
    "historical": (1950, 2014),
    "ssp126":     (2015, 2100),
    "ssp245":     (2015, 2100),
    "ssp370":     (2015, 2100),
    "ssp585":     (2015, 2100),
}

# Rangos válidos para QC (de la Tech Note, Table 3)
VALID_RANGES = {
    "tas":    (200.0, 340.0),   # Kelvin
    "tasmax": (200.0, 340.0),
    "tasmin": (200.0, 340.0),
    "hurs":   (0.0, 102.0),     # Porcentaje
    "huss":   (0.0, 0.04),      # kg/kg
    "pr":     (0.0, 0.012),     # kg/m²/s
}

# =============================================================================
# FUNCIONES
# =============================================================================

def setup_logging(log_file=None):
    """Configura logging a consola y archivo."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def build_s3_url(model, scenario, variable, year, variant):
    """
    Construye la URL de S3 para un archivo NetCDF específico.

    Estructura: NEX-GDDP-CMIP6/{MODEL}/{SCENARIO}/{VARIANT}/{VARIABLE}/
                {var}_day_{MODEL}_{SCENARIO}_{VARIANT}_gn_{YEAR}.nc
    """
    filename = f"{variable}_day_{model}_{scenario}_{variant}_gn_{year}.nc"
    url = f"{S3_BUCKET}/{S3_PREFIX}/{model}/{scenario}/{variant}/{variable}/{filename}"
    return url, filename


def download_file(url, dest_path, max_retries=3):
    """Descarga un archivo con curl, con reintentos."""
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-f", "-L",
                    "--connect-timeout", "30",
                    "--max-time", "600",  # 10 min máximo por archivo
                    "-o", str(dest_path),
                    url,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 1000:
                return True
            else:
                logging.warning(
                    f"  Intento {attempt}/{max_retries} falló (rc={result.returncode}). "
                    f"stderr: {result.stderr[:200]}"
                )
        except Exception as e:
            logging.warning(f"  Intento {attempt}/{max_retries} excepción: {e}")

        if attempt < max_retries:
            time.sleep(5 * attempt)  # Backoff progresivo

    return False


def subset_to_south_america(ds, variable):
    """
    Recorta un dataset xarray a Sudamérica.
    Maneja tanto longitudes 0-360 como -180 a 180.
    """
    lon_values = ds["lon"].values

    # Determinar sistema de coordenadas de longitud
    if lon_values.max() > 180:
        # Sistema 0-360
        lon_min, lon_max = LON_MIN_360, LON_MAX_360
    else:
        # Sistema -180 a 180
        lon_min, lon_max = LON_MIN_180, LON_MAX_180

    # Recortar
    ds_subset = ds.sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(lon_min, lon_max),
    )

    return ds_subset


def validate_data(ds, variable):
    """
    Validación básica de los datos extraídos.
    Retorna dict con métricas de QC.
    """
    data = ds[variable].values
    valid_min, valid_max = VALID_RANGES.get(variable, (None, None))

    total_cells = data.size
    nan_count = np.isnan(data).sum()
    nan_pct = (nan_count / total_cells) * 100

    qc = {
        "total_cells": total_cells,
        "nan_count": int(nan_count),
        "nan_pct": round(nan_pct, 2),
        "data_min": float(np.nanmin(data)) if nan_count < total_cells else None,
        "data_max": float(np.nanmax(data)) if nan_count < total_cells else None,
    }

    if valid_min is not None and qc["data_min"] is not None:
        out_of_range = np.sum((data < valid_min) | (data > valid_max)) - nan_count
        qc["out_of_range"] = max(0, int(out_of_range))
        qc["out_of_range_pct"] = round((qc["out_of_range"] / total_cells) * 100, 4)

    return qc


def process_one_file(model, scenario, variable, year, variant, dry_run=False):
    """
    Procesa un archivo: descarga, recorta, valida, guarda.
    Retorna True si exitoso, False si falló.
    """
    # Construir rutas
    out_dir = RAW_DIR / model / scenario / variable
    out_file = out_dir / f"{variable}_{model}_{scenario}_{year}_SA.nc"

    # Saltar si ya existe
    if out_file.exists():
        logging.debug(f"  Ya existe: {out_file.name}")
        return True

    url, filename = build_s3_url(model, scenario, variable, year, variant)

    if dry_run:
        logging.info(f"  [DRY RUN] {url}")
        return True

    tmp_file = TMP_DIR / filename

    try:
        # 1. Descargar
        t0 = time.time()
        success = download_file(url, tmp_file)
        if not success:
            logging.error(f"  FALLO descarga: {filename}")
            return False
        dl_time = time.time() - t0
        dl_size_mb = tmp_file.stat().st_size / (1024 * 1024)

        # 2. Abrir y recortar
        t1 = time.time()
        ds = xr.open_dataset(tmp_file, engine="netcdf4")
        ds_sa = subset_to_south_america(ds, variable)

        # 3. Validar
        qc = validate_data(ds_sa, variable)

        # 4. Guardar
        out_dir.mkdir(parents=True, exist_ok=True)

        # Configurar encoding para compresión
        encoding = {
            variable: {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
            }
        }
        ds_sa.to_netcdf(out_file, encoding=encoding)
        proc_time = time.time() - t1

        out_size_mb = out_file.stat().st_size / (1024 * 1024)

        ds.close()
        ds_sa.close()

        logging.info(
            f"  OK {filename} | "
            f"dl={dl_time:.0f}s ({dl_size_mb:.0f}MB) | "
            f"proc={proc_time:.1f}s | "
            f"out={out_size_mb:.1f}MB | "
            f"NaN={qc['nan_pct']:.1f}% | "
            f"range=[{qc['data_min']:.1f}, {qc['data_max']:.1f}]"
        )
        return True

    except Exception as e:
        logging.error(f"  ERROR procesando {filename}: {e}")
        return False

    finally:
        # Limpiar archivo temporal
        if tmp_file.exists():
            tmp_file.unlink()


def run_extraction(models, variables, scenarios, dry_run=False):
    """Ejecuta la extracción completa."""
    # Calcular total de archivos
    total_files = 0
    for model in models:
        for scenario, (year_start, year_end) in scenarios.items():
            for variable in variables:
                total_files += (year_end - year_start + 1)

    logging.info(f"Total de archivos a procesar: {total_files}")
    logging.info(f"Modelos: {list(models.keys())}")
    logging.info(f"Variables: {variables}")
    logging.info(f"Escenarios: {list(scenarios.keys())}")
    logging.info("")

    processed = 0
    failed = 0
    skipped = 0
    t_start = time.time()

    for model, variant in models.items():
        for scenario, (year_start, year_end) in scenarios.items():
            for variable in variables:
                logging.info(f"--- {model} / {scenario} / {variable} ({year_start}-{year_end}) ---")

                for year in range(year_start, year_end + 1):
                    out_file = RAW_DIR / model / scenario / variable / \
                               f"{variable}_{model}_{scenario}_{year}_SA.nc"
                    if out_file.exists():
                        skipped += 1
                        continue

                    success = process_one_file(
                        model, scenario, variable, year, variant, dry_run
                    )
                    if success:
                        processed += 1
                    else:
                        failed += 1

                    # Progreso cada 10 archivos
                    done = processed + failed + skipped
                    if done % 10 == 0:
                        elapsed = time.time() - t_start
                        rate = processed / elapsed * 3600 if elapsed > 0 else 0
                        logging.info(
                            f"  Progreso: {done}/{total_files} "
                            f"(ok={processed}, skip={skipped}, fail={failed}) | "
                            f"{rate:.0f} archivos/hora"
                        )

    elapsed = time.time() - t_start
    logging.info("")
    logging.info("=" * 60)
    logging.info(f"  Extracción completada en {elapsed/3600:.1f} horas")
    logging.info(f"  Procesados: {processed}")
    logging.info(f"  Saltados (ya existían): {skipped}")
    logging.info(f"  Fallidos: {failed}")

    # Resumen de almacenamiento
    total_size = sum(
        f.stat().st_size for f in RAW_DIR.rglob("*.nc")
    )
    logging.info(f"  Espacio usado: {total_size / (1024**3):.1f} GB")
    logging.info("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Extracción NEX-GDDP-CMIP6 → Sudamérica")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true",
                      help="Test: 1 modelo, 1 variable, 1 año")
    mode.add_argument("--full", action="store_true",
                      help="Extracción completa: 8 modelos × 2 vars × 5 escenarios")

    parser.add_argument("--model", type=str, help="Modelo específico (ej: ACCESS-CM2)")
    parser.add_argument("--variable", type=str, help="Variable específica (ej: tas)")
    parser.add_argument("--scenario", type=str, help="Escenario específico (ej: historical)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo listar URLs sin descargar")

    args = parser.parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"extraction_{timestamp}.log"
    setup_logging(log_file)

    logging.info("=" * 60)
    logging.info("  NEX-GDDP-CMIP6 → Extracción Sudamérica")
    logging.info(f"  Región: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN_180}, {LON_MAX_180}]")
    logging.info(f"  Log: {log_file}")
    logging.info("=" * 60)

    if args.test:
        # Test mínimo: ACCESS-CM2, tas, historical, 2014 (último año)
        test_models = {"ACCESS-CM2": "r1i1p1f1"}
        test_vars = ["tas"]
        test_scenarios = {"historical": (2014, 2014)}
        logging.info("Modo TEST: ACCESS-CM2 / tas / historical / 2014")
        run_extraction(test_models, test_vars, test_scenarios, args.dry_run)

    elif args.full:
        # Filtrar si se especificó modelo/variable/escenario
        models = MODELS
        variables = VARIABLES
        scenarios = SCENARIOS

        if args.model:
            if args.model not in MODELS:
                logging.error(f"Modelo '{args.model}' no está en la lista. "
                              f"Opciones: {list(MODELS.keys())}")
                sys.exit(1)
            models = {args.model: MODELS[args.model]}

        if args.variable:
            if args.variable not in VARIABLES:
                logging.error(f"Variable '{args.variable}' no está en la lista. "
                              f"Opciones: {VARIABLES}")
                sys.exit(1)
            variables = [args.variable]

        if args.scenario:
            if args.scenario not in SCENARIOS:
                logging.error(f"Escenario '{args.scenario}' no está en la lista. "
                              f"Opciones: {list(SCENARIOS.keys())}")
                sys.exit(1)
            scenarios = {args.scenario: SCENARIOS[args.scenario]}

        run_extraction(models, variables, scenarios, args.dry_run)


if __name__ == "__main__":
    main()
