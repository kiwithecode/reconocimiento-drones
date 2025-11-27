"""
Script mejorado para captura de personas con validación de calidad

Mejoras implementadas:
- Validación de nitidez (sharpness)
- Validación de tamaño mínimo
- Captura inteligente (selecciona mejores frames)
- Diversidad de poses/ángulos
- Feedback visual de calidad
- Prevención de imágenes duplicadas
- Optimizado para captura desde dron
"""

import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from collections import deque
from config import PERSONAS_BASE

# ============================================================================
# CONFIGURACIÓN DE CALIDAD
# ============================================================================

# Calidad de imagen
MIN_SHARPNESS = 100  # Nitidez mínima (Laplacian variance)
MIN_PERSON_SIZE = 50  # Ancho/alto mínimo en pixels
MAX_BLUR_THRESHOLD = 50  # Máximo blur permitido

# Captura inteligente
SIMILARITY_THRESHOLD = 0.85  # Evitar capturas muy similares
DIVERSITY_WINDOW = 10  # Ventana para verificar diversidad
MIN_DISTANCE_BETWEEN_CAPTURES = 1.5  # Segundos entre capturas auto

# Calidad de detección
MIN_DETECTION_CONFIDENCE = 0.6  # Confianza mínima de YOLO

# Recomendaciones de captura
RECOMMENDED_IMAGES = 15  # Número recomendado de imágenes
MINIMUM_IMAGES = 8  # Mínimo absoluto

# ============================================================================
# FUNCIONES DE VALIDACIÓN DE CALIDAD
# ============================================================================

def calculate_sharpness(image):
    """
    Calcula nitidez usando varianza del Laplaciano
    Mayor valor = más nítida la imagen
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def calculate_brightness(image):
    """Calcula brillo promedio de la imagen"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    return np.mean(gray)

def is_image_too_dark(image, threshold=40):
    """Verifica si la imagen está muy oscura"""
    brightness = calculate_brightness(image)
    return brightness < threshold

def is_image_overexposed(image, threshold=220):
    """Verifica si la imagen está sobreexpuesta"""
    brightness = calculate_brightness(image)
    return brightness > threshold

def calculate_image_similarity(img1, img2):
    """
    Calcula similitud entre dos imágenes usando histogramas
    Retorna valor entre 0 (diferentes) y 1 (idénticas)
    """
    if img1 is None or img2 is None:
        return 0.0

    # Redimensionar a mismo tamaño
    size = (128, 256)
    img1_resized = cv2.resize(img1, size)
    img2_resized = cv2.resize(img2, size)

    # Calcular histogramas
    hist1 = cv2.calcHist([img1_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    # Normalizar
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    # Comparar
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def validate_image_quality(image, verbose=False):
    """
    Valida la calidad de una imagen

    Returns:
        tuple: (is_valid, quality_score, reasons)
    """
    reasons = []
    quality_score = 100.0

    # Verificar tamaño
    h, w = image.shape[:2]
    if w < MIN_PERSON_SIZE or h < MIN_PERSON_SIZE:
        reasons.append(f"Muy pequeña ({w}x{h})")
        quality_score -= 40

    # Verificar nitidez
    sharpness = calculate_sharpness(image)
    if sharpness < MIN_SHARPNESS:
        reasons.append(f"Borrosa ({sharpness:.1f})")
        quality_score -= 30
    else:
        # Bonus por nitidez
        quality_score += min(20, (sharpness - MIN_SHARPNESS) / 10)

    # Verificar brillo
    if is_image_too_dark(image):
        reasons.append("Muy oscura")
        quality_score -= 25
    elif is_image_overexposed(image):
        reasons.append("Sobreexpuesta")
        quality_score -= 15

    # Normalizar score
    quality_score = max(0, min(100, quality_score))

    is_valid = quality_score >= 50

    if verbose and not is_valid:
        print(f"❌ Calidad insuficiente: {', '.join(reasons)}")

    return is_valid, quality_score, reasons

# ============================================================================
# CLASE PARA GESTIÓN DE CAPTURAS
# ============================================================================

class CaptureManager:
    """Gestiona la captura inteligente de imágenes"""

    def __init__(self, save_path, person_name):
        self.save_path = save_path
        self.person_name = person_name
        self.counter = len(os.listdir(save_path)) if os.path.exists(save_path) else 0

        # Historial de capturas
        self.recent_captures = deque(maxlen=DIVERSITY_WINDOW)
        self.quality_scores = []

        # Estado
        self.last_auto_capture = 0
        self.auto_mode = False

        # Estadísticas
        self.total_attempts = 0
        self.successful_captures = 0
        self.rejected_blur = 0
        self.rejected_dark = 0
        self.rejected_similar = 0

    def should_capture_auto(self, crop):
        """Decide si debe capturar automáticamente basado en calidad y diversidad"""
        # Verificar tiempo desde última captura
        if time.time() - self.last_auto_capture < MIN_DISTANCE_BETWEEN_CAPTURES:
            return False, "Muy pronto desde última captura"

        # Validar calidad
        is_valid, quality_score, reasons = validate_image_quality(crop)
        if not is_valid:
            return False, f"Calidad baja: {', '.join(reasons)}"

        # Verificar diversidad (que no sea muy similar a capturas recientes)
        if len(self.recent_captures) > 0:
            similarities = [calculate_image_similarity(crop, recent)
                          for recent in self.recent_captures]
            max_similarity = max(similarities)

            if max_similarity > SIMILARITY_THRESHOLD:
                return False, f"Muy similar a captura reciente ({max_similarity:.2f})"

        return True, f"Calidad: {quality_score:.0f}%"

    def save_capture(self, crop, quality_score, mode="manual"):
        """Guarda una captura"""
        self.counter += 1
        filename = f"{self.person_name}_{self.counter:03d}.jpg"
        filepath = os.path.join(self.save_path, filename)

        # Redimensionar a tamaño estándar
        crop_resized = cv2.resize(crop, (128, 256))

        # Guardar
        cv2.imwrite(filepath, crop_resized)

        # Actualizar historial
        self.recent_captures.append(crop.copy())
        self.quality_scores.append(quality_score)

        # Estadísticas
        self.successful_captures += 1

        if mode == "auto":
            self.last_auto_capture = time.time()

        return filename, filepath

    def get_statistics(self):
        """Retorna estadísticas de captura"""
        avg_quality = np.mean(self.quality_scores) if self.quality_scores else 0

        return {
            "total_captures": self.successful_captures,
            "avg_quality": avg_quality,
            "rejected_total": self.total_attempts - self.successful_captures,
            "progress": min(100, (self.successful_captures / RECOMMENDED_IMAGES) * 100)
        }

# ============================================================================
# INTERFAZ VISUAL MEJORADA
# ============================================================================

def draw_quality_indicator(frame, quality_score, x, y):
    """Dibuja indicador visual de calidad"""
    # Determinar color según calidad
    if quality_score >= 80:
        color = (0, 255, 0)  # Verde
        label = "EXCELENTE"
    elif quality_score >= 60:
        color = (0, 200, 255)  # Amarillo
        label = "BUENA"
    elif quality_score >= 40:
        color = (0, 165, 255)  # Naranja
        label = "REGULAR"
    else:
        color = (0, 0, 255)  # Rojo
        label = "MALA"

    # Barra de calidad
    bar_width = 200
    bar_height = 20
    filled_width = int(bar_width * quality_score / 100)

    # Fondo
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (100, 100, 100), -1)

    # Relleno
    cv2.rectangle(frame, (x, y), (x + filled_width, y + bar_height), color, -1)

    # Borde
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)

    # Texto
    text = f"{label} ({quality_score:.0f}%)"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_capture_info(frame, manager, current_quality=None):
    """Dibuja información de captura en pantalla"""
    stats = manager.get_statistics()

    # Panel de información
    panel_y = 50
    line_height = 30

    info_lines = [
        f"Capturas: {stats['total_captures']}/{RECOMMENDED_IMAGES}",
        f"Calidad promedio: {stats['avg_quality']:.0f}%",
        f"Progreso: {stats['progress']:.0f}%",
        f"Modo: {'AUTO' if manager.auto_mode else 'MANUAL'}"
    ]

    for i, line in enumerate(info_lines):
        y = panel_y + i * line_height
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Barra de progreso
    progress_y = panel_y + len(info_lines) * line_height + 10
    progress_width = 300
    progress_filled = int(progress_width * stats['progress'] / 100)

    cv2.rectangle(frame, (10, progress_y), (10 + progress_width, progress_y + 20), (100, 100, 100), -1)
    cv2.rectangle(frame, (10, progress_y), (10 + progress_filled, progress_y + 20), (0, 255, 0), -1)
    cv2.rectangle(frame, (10, progress_y), (10 + progress_width, progress_y + 20), (255, 255, 255), 2)

    # Calidad actual
    if current_quality is not None:
        draw_quality_indicator(frame, current_quality, 10, progress_y + 40)

def draw_instructions(frame):
    """Dibuja instrucciones en pantalla"""
    instructions = [
        "CONTROLES:",
        "'s' - Guardar manual",
        "'a' - Auto ON/OFF",
        "'q' - Salir",
        "",
        "CONSEJOS:",
        "- Varía la pose",
        "- Diferentes ángulos",
        "- Buena iluminación",
        "- Sin objetos tapando"
    ]

    y_start = frame.shape[0] - len(instructions) * 25 - 10

    for i, line in enumerate(instructions):
        y = y_start + i * 25
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("📸 CAPTURA MEJORADA DE PERSONAS - Optimizado para Drones")
    print("="*70)

    # Configuración
    nombre = input("\nIngresa nombre completo: ").strip().replace(" ", "_")
    cedula = input("Ingresa cédula: ").strip()
    person_name = f"{nombre}_{cedula}"

    save_path = os.path.join(PERSONAS_BASE, person_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n✅ Carpeta de guardado: {save_path}")

    # Fuente de video
    print("\nSelecciona fuente de video:")
    print("1. Webcam")
    print("2. Stream RTMP")
    opcion = input("Opción [1/2]: ").strip()

    if opcion == "1":
        source = 0
    elif opcion == "2":
        source = input("URL del stream RTMP: ").strip()
    else:
        print("❌ Opción inválida")
        return

    # Inicializar
    print("\n📦 Cargando modelo YOLO...")
    model = YOLO("yolov8n.pt")
    model.conf = MIN_DETECTION_CONFIDENCE

    print("🎥 Conectando a video...")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("❌ No se pudo abrir la fuente de video")
        return

    # Gestor de capturas
    manager = CaptureManager(save_path, person_name)

    print("\n" + "="*70)
    print("🚀 INICIANDO CAPTURA")
    print("="*70)
    print(f"📊 Recomendación: Capturar al menos {RECOMMENDED_IMAGES} imágenes")
    print(f"⚠️  Mínimo requerido: {MINIMUM_IMAGES} imágenes")
    print("\nControles:")
    print("  's' - Guardar imagen manualmente")
    print("  'a' - Activar/desactivar modo automático")
    print("  'q' - Salir")
    print("="*70 + "\n")

    cv2.namedWindow("Captura Mejorada", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Captura Mejorada", 1280, 720)

    current_quality = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame no recibido")
            break

        # Copia para visualización
        display_frame = frame.copy()

        # Detección
        results = model(frame, verbose=False)

        person_detected = False
        best_crop = None
        best_bbox = None
        best_confidence = 0

        # Encontrar mejor detección
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # persona
                    conf = float(box.conf[0])
                    if conf > best_confidence:
                        best_confidence = conf
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        best_bbox = (x1, y1, x2, y2)
                        best_crop = frame[y1:y2, x1:x2]
                        person_detected = True

        # Procesar mejor detección
        if person_detected and best_crop is not None and best_crop.size > 0:
            x1, y1, x2, y2 = best_bbox

            # Validar calidad
            is_valid, quality_score, reasons = validate_image_quality(best_crop)
            current_quality = quality_score

            # Color del cuadro según calidad
            if quality_score >= 70:
                bbox_color = (0, 255, 0)  # Verde
            elif quality_score >= 50:
                bbox_color = (0, 200, 255)  # Amarillo
            else:
                bbox_color = (0, 0, 255)  # Rojo

            # Dibujar bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), bbox_color, 3)

            # Etiqueta de confianza
            label = f"Conf: {best_confidence:.2f}"
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2)

            # Captura automática
            if manager.auto_mode:
                should_capture, reason = manager.should_capture_auto(best_crop)

                if should_capture:
                    filename, filepath = manager.save_capture(best_crop, quality_score, mode="auto")
                    print(f"📸 AUTO: {filename} - {reason}")
                else:
                    # Mostrar razón de rechazo
                    cv2.putText(display_frame, f"Esperando: {reason}", (x1, y2 + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        else:
            current_quality = None

        # Dibujar interfaz
        draw_capture_info(display_frame, manager, current_quality)
        draw_instructions(display_frame)

        # Mostrar
        cv2.imshow("Captura Mejorada", display_frame)

        # Controles
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s') and best_crop is not None:
            # Captura manual
            is_valid, quality_score, reasons = validate_image_quality(best_crop)

            if is_valid:
                filename, filepath = manager.save_capture(best_crop, quality_score, mode="manual")
                print(f"💾 MANUAL: {filename} - Calidad: {quality_score:.0f}%")
            else:
                print(f"❌ No se puede guardar: {', '.join(reasons)}")

        elif key == ord('a'):
            manager.auto_mode = not manager.auto_mode
            print(f"🔁 Modo automático: {'ACTIVADO' if manager.auto_mode else 'DESACTIVADO'}")

    # Resumen final
    cap.release()
    cv2.destroyAllWindows()

    stats = manager.get_statistics()

    print("\n" + "="*70)
    print("📊 RESUMEN DE CAPTURA")
    print("="*70)
    print(f"✅ Total de imágenes capturadas: {stats['total_captures']}")
    print(f"📈 Calidad promedio: {stats['avg_quality']:.1f}%")
    print(f"📁 Guardadas en: {save_path}")

    if stats['total_captures'] < MINIMUM_IMAGES:
        print(f"\n⚠️  ADVERTENCIA: Menos de {MINIMUM_IMAGES} imágenes")
        print("   Se recomienda capturar más para mejor precisión")
    elif stats['total_captures'] < RECOMMENDED_IMAGES:
        print(f"\n⚡ ACEPTABLE: {stats['total_captures']} imágenes")
        print(f"   Ideal: {RECOMMENDED_IMAGES} para mejor precisión")
    else:
        print(f"\n✨ EXCELENTE: {stats['total_captures']} imágenes capturadas")

    print("\n💡 Siguiente paso:")
    print("   python generar_base_personas_mejorado.py")
    print("="*70)

if __name__ == "__main__":
    main()
