#este codigo es opcional, ya que se puede generar la base de personas con el script generar_base_personas.py

import os
import torch
import torchreid
import numpy as np
import pickle
from PIL import Image
from torchvision import transforms

from config import PERSONAS_BASE, BASE_EMBEDDINGS, MODEL_PATH, DATA_DIR

# Cargar modelo preentrenado de re-identificación
model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)
model.eval()
model.cuda()

# Transformaciones estándar del modelo
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

database = []

# Procesar cada carpeta
for persona in os.listdir(PERSONAS_BASE):
    persona_dir = os.path.join(PERSONAS_BASE, persona)
    if not os.path.isdir(persona_dir):
        continue

    nombre, cedula = persona.rsplit("_", 1)
    embeddings = []

    for img_name in os.listdir(persona_dir):
        img_path = os.path.join(persona_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).cuda()

        with torch.no_grad():
            feature = model(image)
            embeddings.append(feature.cpu().numpy())

    # Promediar embeddings si hay varias imágenes
    mean_embedding = np.mean(np.vstack(embeddings), axis=0)

    database.append({
        "nombre": nombre.replace("_", " ").title(),
        "cedula": cedula,
        "embedding": mean_embedding
    })

def cargar_o_crear_base():
    if os.path.exists(BASE_EMBEDDINGS):
        with open(BASE_EMBEDDINGS, 'rb') as f:
            return pickle.load(f)
    else:
        with open(BASE_EMBEDDINGS, "wb") as f:
            pickle.dump(database, f)
        return database

print("✅ Embeddings generados y guardados en base_personas.pkl")
