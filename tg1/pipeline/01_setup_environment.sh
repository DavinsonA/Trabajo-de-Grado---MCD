#!/bin/bash
# ==============================================================================
# 01_setup_environment.sh
# Setup del entorno para el proyecto NEX-GDDP-CMIP6
# Ejecutar en la máquina remota: bash 01_setup_environment.sh
# ==============================================================================

set -e  # Detener si hay errores

echo "============================================"
echo "  Setup del entorno - NEX-GDDP-CMIP6"
echo "============================================"

# --- 1. Crear estructura de directorios en el volumen ---
echo ""
echo "[1/5] Creando estructura de directorios..."

DATA_DIR="/media/volume/jay2vol/nexgddp"
mkdir -p "${DATA_DIR}/raw"          # NetCDF recortados a Sudamérica
mkdir -p "${DATA_DIR}/processed"    # Datos procesados (anomalías, etc.)
mkdir -p "${DATA_DIR}/models"       # Modelos ML entrenados
mkdir -p "${DATA_DIR}/outputs"      # Visualizaciones, dashboards
mkdir -p "${DATA_DIR}/tmp"          # Archivos temporales (descargas globales)
mkdir -p "${DATA_DIR}/logs"         # Logs de extracción

echo "  Directorios creados en ${DATA_DIR}"
ls -la "${DATA_DIR}"

# --- 2. Instalar dependencias del sistema ---
echo ""
echo "[2/5] Instalando dependencias del sistema..."

sudo apt-get update -qq
sudo apt-get install -y -qq \
    libhdf5-dev \
    libnetcdf-dev \
    libgeos-dev \
    libproj-dev \
    build-essential \
    curl \
    wget \
    tmux \
    htop \
    > /dev/null 2>&1

echo "  Dependencias del sistema instaladas."

# --- 3. Crear entorno virtual de Python ---
echo ""
echo "[3/5] Creando entorno virtual de Python..."

VENV_DIR="${DATA_DIR}/venv"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "  Entorno virtual creado en ${VENV_DIR}"
echo "  Python: $(python --version)"

# --- 4. Instalar paquetes de Python ---
echo ""
echo "[4/5] Instalando paquetes de Python (esto toma ~5 minutos)..."

pip install --upgrade pip setuptools wheel -q

# Core data
pip install -q \
    numpy \
    pandas \
    scipy

# Climate/NetCDF
pip install -q \
    xarray \
    netCDF4 \
    h5netcdf \
    dask[complete] \
    zarr \
    cftime \
    bottleneck

# Visualization
pip install -q \
    matplotlib \
    hvplot \
    holoviews \
    panel \
    bokeh \
    cartopy \
    cmocean

# ML (secundario, instalar ahora para evitar reinstalaciones)
pip install -q \
    scikit-learn \
    xgboost \
    shap

# Statistics
pip install -q \
    pymannkendall \
    statsmodels

# OpenVisus (para consultas puntuales)
pip install -q OpenVisus

# Jupyter (para notebooks locales si se necesitan)
pip install -q \
    jupyter \
    ipykernel

echo "  Paquetes de Python instalados."

# --- 5. Verificación ---
echo ""
echo "[5/5] Verificando instalación..."
echo ""

python -c "
import xarray; print(f'  xarray      {xarray.__version__}')
import dask; print(f'  dask        {dask.__version__}')
import netCDF4; print(f'  netCDF4     {netCDF4.__version__}')
import zarr; print(f'  zarr        {zarr.__version__}')
import numpy; print(f'  numpy       {numpy.__version__}')
import pandas; print(f'  pandas      {pandas.__version__}')
import matplotlib; print(f'  matplotlib  {matplotlib.__version__}')
import cartopy; print(f'  cartopy     {cartopy.__version__}')
import sklearn; print(f'  scikit-learn {sklearn.__version__}')
import xgboost; print(f'  xgboost     {xgboost.__version__}')
import shap; print(f'  shap        {shap.__version__}')
"

echo ""
echo "============================================"
echo "  Setup completado exitosamente!"
echo ""
echo "  Para activar el entorno:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "  Datos se almacenarán en:"
echo "  ${DATA_DIR}"
echo "============================================"
