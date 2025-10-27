import os
import cv2
import torch
import pickle
import numpy as np
from torchreid.utils import FeatureExtractor
from config import PERSONAS_BASE, BASE_EMBEDDINGS, MODEL_PATH, DATA_DIR

# Cargar el extractor de características con el modelo preentrenado
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=MODEL_PATH,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

base_embeddings = []

# Recorrer cada subcarpeta
for persona in os.listdir(PERSONAS_BASE):
    persona_dir = os.path.join(PERSONAS_BASE, persona)
    if not os.path.isdir(persona_dir):
        continue

    # Parsear nombre y cédula desde el nombre de la carpeta
    partes = persona.split("_")
    if len(partes) >= 3:
        cedula = partes[-1]
        nombre = " ".join(partes[:-1]).replace("_", " ").title()
    else:
        nombre = persona
        cedula = "0000000000"  # fallback por si no se encuentra

    embeddings = []

    for archivo in os.listdir(persona_dir):
        if not archivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        ruta_imagen = os.path.join(persona_dir, archivo)
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
        print(f"⚠️ No se encontraron imágenes válidas para {persona}.")

# Guardar el archivo .pkl con todos los embeddings
with open(BASE_EMBEDDINGS, 'wb') as f:
    pickle.dump(base_embeddings, f)

print(f"\n✅ Base de {len(base_embeddings)} personas guardada en {BASE_EMBEDDINGS}")
