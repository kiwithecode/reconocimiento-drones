import os
import cv2
from ultralytics import YOLO
import torch
import pickle
import numpy as np
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import time
from config import DATA_DIR, BASE_EMBEDDINGS

# Configuración de parámetros para máxima velocidad
FRAME_SKIP = 3  # Procesar 1 de cada 3 frames (aumentado para más velocidad)
SCALE_PERCENT = 40  # Reducción de resolución al 40% (más rápido pero menos preciso)
MIN_CONFIDENCE = 0.6  # Umbral de confianza más alto para menos detecciones
UPDATE_INTERVAL = 8  # Actualizar identificación cada 8 frames

# Reducir el tamaño de la ventana de visualización
cv2.namedWindow('Identificación de Personas', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Identificación de Personas', 800, 600)  # Tamaño fijo más pequeño

# Cargar modelo YOLO (versión más ligera)
model = YOLO("yolov8n.pt")  # Cambiado a 'nano' para mejor rendimiento
model.conf = MIN_CONFIDENCE
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar extractor de características (usando el modelo que entrenaste)
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path=os.path.join(DATA_DIR, 'osnet_x1_0_imagenet.pth'),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Cargar base de embeddings
with open(BASE_EMBEDDINGS, "rb") as f:
    base = pickle.load(f)

# Stream RTMP
rtmp_url = "rtmp://localhost:1935/live/stream"
cap = cv2.VideoCapture(rtmp_url)

if not cap.isOpened():
    print("❌ No se pudo abrir el stream RTMP.")
    exit()

print("✅ Stream conectado. Iniciando re-identificación...")

def identificar(embedding, base, threshold=0.8):
    if not base:  # Si la base está vacía
        return "Desconocido"
        
    # Convertir a numpy una sola vez
    emb_np = embedding.cpu().numpy()
    
    # Precalcular todos los embeddings para comparación por lotes
    base_embs = np.array([p["embedding"] for p in base])
    
    # Calcular similitudes en lote
    similarities = cosine_similarity([emb_np], base_embs)[0]
    
    # Encontrar la mejor coincidencia
    max_idx = np.argmax(similarities)
    mejor_score = similarities[max_idx]
    
    if mejor_score >= threshold:
        return f'{base[max_idx]["nombre"]} - {mejor_score:.2f}'
    return "Desconocido"

# Inicializar variables para el seguimiento
tracked_objects = {}
frame_count = 0
last_fps_time = time.time()
fps_counter = 0

while True:
    # Capturar frame
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame no recibido.")
        break
    
    frame_count += 1
    
    # Reducir resolución para procesamiento
    if SCALE_PERCENT != 100:
        width = int(frame.shape[1] * SCALE_PERCENT / 100)
        height = int(frame.shape[0] * SCALE_PRECENT / 100)
        frame_small = cv2.resize(frame, (width, height))
    else:
        frame_small = frame
        
    # Usar la versión pequeña para detección
    frame_detection = frame_small
    
    # Procesar solo algunos frames
    if frame_count % FRAME_SKIP != 0:
        cv2.imshow("Identificación de Personas", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    
    # Detección con YOLO en la versión pequeña
    results = model(frame_detection, verbose=False, imgsz=320)  # Tamaño de imagen fijo para consistencia
    
    # Procesar detecciones
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                if conf < MIN_CONFIDENCE:
                    continue
                
                # Obtener región de interés (escalar coordenadas si usamos frame reducido)
                if SCALE_PRECENT != 100:
                    scale = 100 / SCALE_PRECENT
                    x1_orig, y1_orig = int(x1 * scale), int(y1 * scale)
                    x2_orig, y2_orig = int(x2 * scale), int(y2 * scale)
                    crop = frame[y1_orig:y2_orig, x1_orig:x2_orig]
                else:
                    crop = frame[y1:y2, x1:x2]
                    
                if crop.size == 0:
                    continue
                
                # Redimensionar al mínimo necesario para el extractor
                crop_resized = cv2.resize(crop, (96, 192))  # Tamaño reducido
                
                # Actualizar identificación solo cada cierto tiempo
                obj_id = f"{x1}_{y1}_{x2}_{y2}"  # ID simple basado en posición
                
                if obj_id not in tracked_objects or frame_count % UPDATE_INTERVAL == 0:
                    try:
                        emb = extractor(crop_resized)[0]
                        nombre = identificar(emb, base)
                        tracked_objects[obj_id] = {
                            'nombre': nombre,
                            'bbox': (x1, y1, x2, y2)
                        }
                    except Exception as e:
                        print(f"Error en extracción de características: {e}")
                        continue
                
                # Dibujar caja y nombre
                if obj_id in tracked_objects:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, 
                              tracked_objects[obj_id]['nombre'], 
                              (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, 
                              (0, 255, 0), 
                              2)
    
    # Calcular y mostrar FPS
    fps_counter += 1
    if time.time() - last_fps_time >= 1.0:  # Actualizar FPS cada segundo
        fps = fps_counter / (time.time() - last_fps_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        last_fps_time = time.time()
        fps_counter = 0
    
    # Mostrar frame
    cv2.imshow("Identificación de Personas", frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
