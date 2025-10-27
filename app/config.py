import os

# Rutas base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Rutas específicas
PERSONAS_BASE = os.path.join(DATA_DIR, 'personas_base')
BASE_EMBEDDINGS = os.path.join(DATA_DIR, 'base_personas.pkl')
MODEL_PATH = os.path.join(DATA_DIR, 'osnet_x1_0_imagenet.pth')

# Crear directorios si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PERSONAS_BASE, exist_ok=True)
