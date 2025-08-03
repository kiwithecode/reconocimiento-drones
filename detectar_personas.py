import cv2
from ultralytics import YOLO

# Carga el modelo YOLOv8. Usa yolov8n.pt (nano) si no tienes GPU.
model = YOLO("yolov8n.pt")  # También puedes usar yolov8s.pt o yolov8m.pt

# URL del stream RTMP (ya confirmado)
rtmp_url = "rtmp://localhost:1935/live/stream"

# Captura el stream
cap = cv2.VideoCapture(rtmp_url)

if not cap.isOpened():
    print("❌ No se pudo abrir el stream RTMP.")
    exit()

print("✅ Stream conectado. Iniciando detección...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame no recibido.")
        break

    # Ejecutar YOLO (detecta por defecto todas las clases del COCO dataset)
    results = model(frame)

    # Filtrar solo personas (clase 0 en COCO)
    for r in results:
        r.boxes = [box for box in r.boxes if int(box.cls[0]) == 0]

    # Dibujar los resultados en el frame
    annotated_frame = results[0].plot()

    # Mostrar el video con detección
    cv2.imshow("YOLOv8 - Personas detectadas", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
