import cv2
import os
import time
from ultralytics import YOLO

# Configuración
nombre_persona = "kevin_armas_1726414087"
carpeta_base = "personas_base"
ruta_guardado = os.path.join(carpeta_base, nombre_persona)
os.makedirs(ruta_guardado, exist_ok=True)

# Preguntar por fuente de video
print("Selecciona la fuente de video:")
print("1. Cámara Web")
print("2. Stream RTMP")
opcion = input("Opción [1/2]: ").strip()

if opcion == "1":
    fuente = 0
elif opcion == "2":
    fuente = input("Ingresa la URL del stream RTMP: ").strip()
else:
    print("❌ Opción inválida.")
    exit()

# Inicializar modelo YOLO
model = YOLO("yolov8n.pt")

# Inicializar captura de video
cap = cv2.VideoCapture(fuente)

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara o stream.")
    exit()

print("✅ Cámara activa.")
print("Presiona 's' para guardar una imagen manualmente.")
print("Presiona 'a' para activar/desactivar guardado automático cada 2 segundos.")
print("Presiona 'q' para salir.\n")

# Estado
contador = len(os.listdir(ruta_guardado))
auto_guardado = False
ultimo_guardado = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame no recibido.")
        break

    results = model(frame)
    persona_detectada = False
    crop_actual = None

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # clase persona
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                crop_actual = frame[y1:y2, x1:x2]
                persona_detectada = True

    if persona_detectada and crop_actual is not None:
        # Auto guardado cada 2 segundos
        if auto_guardado and (time.time() - ultimo_guardado > 2):
            contador += 1
            nombre_archivo = f"{nombre_persona}_{contador}.jpg"
            ruta_img = os.path.join(ruta_guardado, nombre_archivo)
            crop_resized = cv2.resize(crop_actual, (128, 256))
            cv2.imwrite(ruta_img, crop_resized)
            print(f"🧠 Imagen automática guardada: {nombre_archivo}")
            ultimo_guardado = time.time()

        # Mostrar texto en pantalla
        cv2.putText(frame, f"Persona detectada - AutoGuardado: {'ON' if auto_guardado else 'OFF'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "🛑 No se detecta persona", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Captura de Persona", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s") and crop_actual is not None:
        contador += 1
        nombre_archivo = f"{nombre_persona}_{contador}.jpg"
        ruta_img = os.path.join(ruta_guardado, nombre_archivo)
        crop_resized = cv2.resize(crop_actual, (128, 256))
        cv2.imwrite(ruta_img, crop_resized)
        print(f"💾 Imagen manual guardada: {nombre_archivo}")
    elif key == ord("a"):
        auto_guardado = not auto_guardado
        print(f"🔁 AutoGuardado {'activado' if auto_guardado else 'desactivado'}")

cap.release()
cv2.destroyAllWindows()
