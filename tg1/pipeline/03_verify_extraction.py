#!/usr/bin/env python3
"""
03_verify_extraction.py
Verifica que la extracción de prueba fue correcta.

Uso:
    source /media/volume/jay2vol/nexgddp/venv/bin/activate
    python 03_verify_extraction.py
"""

import xarray as xr
import numpy as np
from pathlib import Path

RAW_DIR = Path("/media/volume/jay2vol/nexgddp/raw")

def verify():
    print("=" * 60)
    print("  Verificación de extracción NEX-GDDP-CMIP6")
    print("=" * 60)

    # Buscar todos los archivos extraídos
    nc_files = sorted(RAW_DIR.rglob("*.nc"))
    print(f"\nArchivos encontrados: {len(nc_files)}")

    if not nc_files:
        print("  ¡No se encontraron archivos! Ejecuta primero 02_extract_nexgddp.py --test")
        return

    for f in nc_files:
        print(f"\n--- {f.relative_to(RAW_DIR)} ---")
        print(f"  Tamaño: {f.stat().st_size / (1024*1024):.1f} MB")

        ds = xr.open_dataset(f)

        # Dimensiones
        print(f"  Dimensiones: {dict(ds.dims)}")
        print(f"  Variables: {list(ds.data_vars)}")

        # Coordenadas
        lat = ds["lat"].values
        lon = ds["lon"].values
        print(f"  Lat range: [{lat.min():.2f}, {lat.max():.2f}] ({len(lat)} puntos)")
        print(f"  Lon range: [{lon.min():.2f}, {lon.max():.2f}] ({len(lon)} puntos)")

        # Verificar que es Sudamérica
        if lat.min() >= -57 and lat.max() <= 14:
            print(f"  ✓ Recorte de latitud correcto (Sudamérica)")
        else:
            print(f"  ✗ LATITUD FUERA DE RANGO - revisar subsetting")

        # Tiempo
        if "time" in ds.dims:
            time_vals = ds["time"].values
            print(f"  Tiempo: {str(time_vals[0])[:10]} a {str(time_vals[-1])[:10]} ({len(time_vals)} días)")

        # Datos
        for var in ds.data_vars:
            data = ds[var].values
            nan_pct = np.isnan(data).sum() / data.size * 100
            print(f"  Variable '{var}':")
            print(f"    Min: {np.nanmin(data):.2f}")
            print(f"    Max: {np.nanmax(data):.2f}")
            print(f"    Mean: {np.nanmean(data):.2f}")
            print(f"    NaN: {nan_pct:.1f}%")

            # Validar rangos físicos
            if var == "tas":
                if 200 <= np.nanmin(data) and np.nanmax(data) <= 340:
                    print(f"    ✓ Rango de temperatura válido (200-340 K)")
                else:
                    print(f"    ✗ TEMPERATURA FUERA DE RANGO FÍSICO")
            elif var == "hurs":
                if 0 <= np.nanmin(data) and np.nanmax(data) <= 102:
                    print(f"    ✓ Rango de humedad relativa válido (0-102%)")
                else:
                    print(f"    ✗ HUMEDAD FUERA DE RANGO FÍSICO")

        ds.close()

    # Resumen de almacenamiento
    total_size = sum(f.stat().st_size for f in nc_files)
    print(f"\n{'=' * 60}")
    print(f"  Total archivos: {len(nc_files)}")
    print(f"  Espacio total: {total_size / (1024**3):.2f} GB")
    print(f"  Promedio por archivo: {total_size / len(nc_files) / (1024**2):.1f} MB")

    # Estimación para extracción completa
    # 8 modelos × 2 vars × (65 hist + 4×86 ssp) = 8 × 2 × 409 = 6,544 archivos
    est_total = 6544
    est_size_gb = (total_size / len(nc_files)) * est_total / (1024**3)
    print(f"\n  Estimación extracción completa:")
    print(f"    Archivos: ~{est_total}")
    print(f"    Espacio estimado: ~{est_size_gb:.0f} GB")
    print(f"=" * 60)


if __name__ == "__main__":
    verify()
