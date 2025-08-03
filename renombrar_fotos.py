import os

# === DATOS A PERSONALIZAR ===
nombre = "Kevin_Armas"
cedula = "1726414087"
carpeta_origen = "personas_base/kevin_armas_1726414087"  # Carpeta donde están las subcarpetas de personas

# === Renombrar imágenes ===
imagenes = [f for f in os.listdir(carpeta_origen) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
imagenes.sort()  # Ordena para mantener secuencia

for i, archivo in enumerate(imagenes, start=1):
    extension = os.path.splitext(archivo)[1]
    nuevo_nombre = f"{cedula}_{nombre}_{i}{extension}"
    ruta_anterior = os.path.join(carpeta_origen, archivo)
    ruta_nueva = os.path.join(carpeta_origen, nuevo_nombre)

    os.rename(ruta_anterior, ruta_nueva)
    print(f"✅ Renombrado: {archivo} → {nuevo_nombre}")

print("\n✅ Todos los archivos han sido renombrados.")
