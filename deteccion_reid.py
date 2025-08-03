import cv2
from ultralytics import YOLO
import torch
import pickle
import numpy as np
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo YOLO
model = YOLO("yolov8m.pt")

# Cargar extractor de características (usando el modelo que entrenaste)
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='osnet_x1_0_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Cargar base de embeddings
with open("base_personas.pkl", "rb") as f:
    base = pickle.load(f)

# Stream RTMP
rtmp_url = "rtmp://localhost:1935/live/stream"
cap = cv2.VideoCapture(rtmp_url)

if not cap.isOpened():
    print("❌ No se pudo abrir el stream RTMP.")
    exit()

print("✅ Stream conectado. Iniciando re-identificación...")

def identificar(embedding, base, threshold=0.8):
    mejor_match = None
    mejor_score = 0
    for persona in base:
        sim = cosine_similarity(
            [embedding.cpu().numpy()],
            [persona["embedding"]]
        )[0][0]
        if sim > mejor_score:
            mejor_score = sim
            mejor_match = persona
    if mejor_score >= threshold:
        return f'{mejor_match["nombre"]} - {mejor_score:.2f}'
    else:
        return "Desconocido"

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame no recibido.")
        break

    # Detección con YOLO
    results = model(frame)
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]

                # Redimensionar al tamaño esperado (256x128)
                crop_resized = cv2.resize(crop, (128, 256))
                emb = extractor(crop_resized)[0]

                nombre = identificar(emb, base)

                # Dibujar caja y nombre
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, nombre, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Identificación de Personas", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
