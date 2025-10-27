import os
import re
from config import PERSONAS_BASE as carpeta_base

# === DATOS A PERSONALIZAR ===
nombre = "Kevin_Armas"
cedula = "1726414087"

# === Renombrar imágenes ===
for persona in os.listdir(carpeta_base):
    persona_dir = os.path.join(carpeta_base, persona)
    imagenes = [f for f in os.listdir(persona_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    imagenes.sort()  # Ordena para mantener secuencia

    for i, archivo in enumerate(imagenes, start=1):
        extension = os.path.splitext(archivo)[1]
        nuevo_nombre = f"{cedula}_{nombre}_{i}{extension}"
        ruta_anterior = os.path.join(persona_dir, archivo)
        ruta_nueva = os.path.join(persona_dir, nuevo_nombre)

        os.rename(ruta_anterior, ruta_nueva)
        print(f"✅ Renombrado: {archivo} → {nuevo_nombre}")

print("\n✅ Todos los archivos han sido renombrados.")
