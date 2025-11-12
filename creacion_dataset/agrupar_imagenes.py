from astropy.io import fits
import numpy as np
import glob
import re
import os

# === CONFIGURACIÓN ===
ruta = r"D:\FIE\Photsat\photsat_frames_cosmic_rays"
patron = os.path.join(ruta, "Img_steady_capture_*_scanning__optic_1_0.fits")
salida = os.path.join(ruta, "sumadas")  # carpeta donde se guardarán las sumas
os.makedirs(salida, exist_ok=True)

# === 1. Obtener lista de archivos ===
archivos = sorted(glob.glob(patron))

print(f"Se encontraron {len(archivos)} archivos FITS.")

# === 2. Extraer número de bloque (ej: 2200, 2201, etc.) usando regex ===
def extraer_bloque(nombre):
    """
    Extrae el número de bloque de nombres tipo:
    Img_steady_capture_35300_2200_scanning__optic_0_0.fits
    """
    m = re.search(r"capture_\d+_(\d+)_scanning", nombre)
    return int(m.group(1)) if m else None

# Agrupar archivos por bloque
bloques = {}
for archivo in archivos:
    bloque = extraer_bloque(archivo)
    if bloque is None:
        continue
    bloques.setdefault(bloque, []).append(archivo)

# === 3. Procesar cada bloque (sumar cada grupo de 15 imágenes) ===
for bloque, lista in sorted(bloques.items()):
    lista = sorted(lista)
    if len(lista) < 15:
        print(f"Bloque {bloque} tiene solo {len(lista)} archivos, se omite.")
        continue

    print(f"Procesando bloque {bloque} ({len(lista)} archivos)...")

    suma = None
    for archivo in lista[:15]:  # solo las primeras 15 del grupo
        with fits.open(archivo) as hdul:
            data = hdul[0].data.astype(float)
            if suma is None:
                suma = np.zeros_like(data)
            suma += data

    # === 4. Guardar resultado ===
    nombre_salida = os.path.join(salida, f"suma_{bloque}_1_0.fits")
    fits.PrimaryHDU(suma).writeto(nombre_salida, overwrite=True)

    print(f"Guardado: {nombre_salida}")

print("Proceso completado.")

