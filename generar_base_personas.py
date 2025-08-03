import os
import cv2
import torch
import pickle
import numpy as np
from torchreid.utils import FeatureExtractor

# Carpeta donde están las subcarpetas de personas
carpeta_personas = "personas_base"
output_pkl = "base_personas.pkl"

# Cargar el extractor de características con el modelo preentrenado
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='osnet_x1_0_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

base_embeddings = []

# Recorrer cada subcarpeta
for subcarpeta in os.listdir(carpeta_personas):
    ruta_subcarpeta = os.path.join(carpeta_personas, subcarpeta)
    if not os.path.isdir(ruta_subcarpeta):
        continue

    # Parsear nombre y cédula desde el nombre de la carpeta
    partes = subcarpeta.split("_")
    if len(partes) >= 3:
        cedula = partes[-1]
        nombre = " ".join(partes[:-1]).replace("_", " ").title()
    else:
        nombre = subcarpeta
        cedula = "0000000000"  # fallback por si no se encuentra

    embeddings = []

    for archivo in os.listdir(ruta_subcarpeta):
        if not archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        ruta_imagen = os.path.join(ruta_subcarpeta, archivo)
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"⚠️ No se pudo leer: {ruta_imagen}")
            continue

        imagen = cv2.resize(imagen, (128, 256))
        emb = extractor(imagen)[0].cpu().numpy()
        embeddings.append(emb)

    if embeddings:
        emb_promedio = np.mean(embeddings, axis=0)
        base_embeddings.append({
            "nombre": nombre,
            "cedula": cedula,
            "embedding": emb_promedio
        })
        print(f"✅ {nombre} ({cedula}) procesado con {len(embeddings)} imágenes.")
    else:
        print(f"⚠️ No se encontraron imágenes válidas para {subcarpeta}.")

# Guardar el archivo .pkl con todos los embeddings
with open(output_pkl, "wb") as f:
    pickle.dump(base_embeddings, f)

print(f"\n✅ Base de {len(base_embeddings)} personas guardada en {output_pkl}")
