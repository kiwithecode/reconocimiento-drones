"""
Configuración optimizada para Re-Identificación desde Dron DJI FPV

Este archivo permite ajustar fácilmente los parámetros del sistema
según las condiciones de vuelo y calidad del video del dron.
"""

# ============================================================================
# CONFIGURACIÓN DE VIDEO
# ============================================================================

class VideoConfig:
    """Configuración de fuente de video"""

    # Fuente de video (opciones):
    # - "rtmp://localhost:1935/live/stream"  # Stream RTMP
    # - 0  # Webcam
    # - "ruta/al/video.mp4"  # Archivo de video
    # - "udp://192.168.1.100:8888"  # Stream UDP desde DJI
    VIDEO_SOURCE = "rtmp://localhost:1935/live/stream"

    # Resolución de procesamiento (% del original)
    # Mayor = mejor calidad pero más lento
    # Recomendado: 50-70 para drones
    SCALE_PERCENT = 60

    # Frames a saltar (1 = procesar todos, 2 = 1 de cada 2, etc.)
    # Para drones en movimiento, se recomienda 1 o 2
    FRAME_SKIP = 1


# ============================================================================
# CONFIGURACIÓN DE DETECCIÓN (YOLO)
# ============================================================================

class DetectionConfig:
    """Configuración de detección de personas"""

    # Modelo YOLO a usar
    # Opciones: "yolov8n.pt" (rápido), "yolov8s.pt", "yolov8m.pt" (preciso)
    YOLO_MODEL = "yolov8n.pt"

    # Confianza mínima para detección (0.0 - 1.0)
    # Más bajo = más detecciones pero más falsos positivos
    # Recomendado para drones: 0.3 - 0.5
    MIN_CONFIDENCE = 0.4

    # Tamaño de imagen para YOLO
    # Opciones: 320 (rápido), 640 (balanced), 1280 (preciso)
    IMGSZ = 640


# ============================================================================
# CONFIGURACIÓN DE RE-IDENTIFICACIÓN
# ============================================================================

class ReIDConfig:
    """Configuración de re-identificación"""

    # Umbral de similitud para identificación (0.0 - 1.0)
    # Más alto = más estricto, menos falsos positivos
    # Recomendado para drones: 0.50 - 0.60 (más permisivo por ángulos variables)
    IDENTIFICATION_THRESHOLD = 0.55

    # Cada cuántos frames actualizar la identificación
    # Mayor = más estable pero menos responsive
    # Recomendado: 3-7 para drones
    REID_UPDATE_INTERVAL = 5

    # Tamaño de redimensión para extractor de features
    # (width, height) - NO cambiar a menos que sepas lo que haces
    REID_INPUT_SIZE = (128, 256)


# ============================================================================
# CONFIGURACIÓN DE TRACKING
# ============================================================================

class TrackingConfig:
    """Configuración de tracking de personas"""

    # Umbral de IoU para asociar detecciones (0.0 - 1.0)
    # Más bajo = más permisivo con movimiento
    # Recomendado para drones: 0.2 - 0.4 (el dron se mueve mucho)
    IOU_THRESHOLD = 0.3

    # Frames que se mantiene el track después de perder detección
    # Mayor = cuadros permanecen más tiempo (menos parpadeo)
    # Recomendado: 10-20 para drones
    TRACK_PERSISTENCE_FRAMES = 15

    # Máximo de frames sin detección antes de eliminar el track
    # Recomendado: 20-40 para drones
    MAX_TRACK_AGE = 30

    # Mínimo de detecciones para confirmar un track
    # Evita falsos positivos
    MIN_DETECTIONS_TO_CONFIRM = 3


# ============================================================================
# CONFIGURACIÓN DE SUAVIZADO
# ============================================================================

class SmoothingConfig:
    """Configuración de suavizado de detecciones"""

    # Ventana de suavizado de bounding boxes
    # Mayor = movimiento más suave pero menos responsive
    # Recomendado para drones: 3-7 (compensa vibraciones)
    SMOOTHING_WINDOW = 5

    # Ventana para votación de identidad
    # Número de frames para decidir identidad por votación
    # Mayor = más estable pero menos responsive
    VOTE_WINDOW = 3


# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIÓN
# ============================================================================

class DisplayConfig:
    """Configuración de visualización"""

    # Tamaño de ventana (width, height)
    WINDOW_SIZE = (1280, 720)

    # Grosor de bounding boxes
    BBOX_THICKNESS = 3

    # Escala de fuente
    FONT_SCALE = 0.8

    # Mostrar barra de confianza
    SHOW_CONFIDENCE_BAR = True

    # Ancho de barra de confianza (pixels)
    CONFIDENCE_BAR_WIDTH = 100

    # Mostrar estadísticas en pantalla
    SHOW_STATS = True


# ============================================================================
# PRESETS PARA DIFERENTES ESCENARIOS
# ============================================================================

class Presets:
    """Presets predefinidos para diferentes escenarios"""

    @staticmethod
    def apply_high_quality():
        """Máxima calidad - para drones estacionarios o vuelo lento"""
        VideoConfig.SCALE_PERCENT = 80
        VideoConfig.FRAME_SKIP = 1
        DetectionConfig.MIN_CONFIDENCE = 0.5
        DetectionConfig.IMGSZ = 640
        ReIDConfig.IDENTIFICATION_THRESHOLD = 0.60
        ReIDConfig.REID_UPDATE_INTERVAL = 3
        TrackingConfig.IOU_THRESHOLD = 0.4
        TrackingConfig.TRACK_PERSISTENCE_FRAMES = 10
        print("✅ Preset aplicado: ALTA CALIDAD")

    @staticmethod
    def apply_balanced():
        """Balance calidad/velocidad - para vuelo normal"""
        VideoConfig.SCALE_PERCENT = 60
        VideoConfig.FRAME_SKIP = 1
        DetectionConfig.MIN_CONFIDENCE = 0.4
        DetectionConfig.IMGSZ = 640
        ReIDConfig.IDENTIFICATION_THRESHOLD = 0.55
        ReIDConfig.REID_UPDATE_INTERVAL = 5
        TrackingConfig.IOU_THRESHOLD = 0.3
        TrackingConfig.TRACK_PERSISTENCE_FRAMES = 15
        print("✅ Preset aplicado: BALANCEADO (recomendado)")

    @staticmethod
    def apply_high_speed():
        """Máxima velocidad - para drones rápidos o hardware limitado"""
        VideoConfig.SCALE_PERCENT = 40
        VideoConfig.FRAME_SKIP = 2
        DetectionConfig.MIN_CONFIDENCE = 0.3
        DetectionConfig.IMGSZ = 320
        ReIDConfig.IDENTIFICATION_THRESHOLD = 0.50
        ReIDConfig.REID_UPDATE_INTERVAL = 7
        TrackingConfig.IOU_THRESHOLD = 0.25
        TrackingConfig.TRACK_PERSISTENCE_FRAMES = 20
        print("✅ Preset aplicado: ALTA VELOCIDAD")

    @staticmethod
    def apply_fpv_racing():
        """Optimizado para FPV racing - movimiento muy rápido"""
        VideoConfig.SCALE_PERCENT = 50
        VideoConfig.FRAME_SKIP = 2
        DetectionConfig.MIN_CONFIDENCE = 0.35
        DetectionConfig.IMGSZ = 320
        ReIDConfig.IDENTIFICATION_THRESHOLD = 0.50
        ReIDConfig.REID_UPDATE_INTERVAL = 8
        TrackingConfig.IOU_THRESHOLD = 0.2
        TrackingConfig.TRACK_PERSISTENCE_FRAMES = 25
        TrackingConfig.MAX_TRACK_AGE = 40
        SmoothingConfig.SMOOTHING_WINDOW = 7
        print("✅ Preset aplicado: FPV RACING")


# ============================================================================
# APLICAR PRESET POR DEFECTO
# ============================================================================

# Descomentar el preset que quieras usar:
Presets.apply_balanced()  # ← Preset por defecto
# Presets.apply_high_quality()
# Presets.apply_high_speed()
# Presets.apply_fpv_racing()


# ============================================================================
# CONFIGURACIÓN DE CONTEO DE PERSONAS
# ============================================================================

class CountingConfig:
    """Configuración del sistema de conteo de personas"""

    # Activar sistema de conteo
    ENABLE_COUNTING = True

    # Contar solo personas únicas (evitar contar la misma persona múltiples veces)
    COUNT_UNIQUE_ONLY = True

    # Intervalo de guardado automático de estadísticas (segundos)
    # 0 = desactivar guardado automático
    SAVE_STATS_INTERVAL = 60

    # Formato de exportación: "csv", "json", "both"
    EXPORT_FORMAT = "both"

    # Directorio donde guardar estadísticas
    # None = usar DATA_DIR/estadisticas
    STATS_OUTPUT_DIR = None

    # Mostrar panel de conteo en pantalla
    SHOW_COUNT_PANEL = True

    # Posición del panel: "right", "left"
    PANEL_POSITION = "right"

    # Tamaño del panel
    PANEL_WIDTH = 350
    PANEL_HEIGHT = 350


# ============================================================================
# CONFIGURACIÓN DE ZONAS DE CONTEO
# ============================================================================

class ZonesConfig:
    """Configuración de zonas de interés para conteo"""

    # Activar sistema de zonas
    ENABLE_ZONES = False

    # Definición de zonas (coordenadas en pixels)
    # Formato: {"nombre": {"x1": x, "y1": y, "x2": x, "y2": y}}
    ZONES = {
        "zona_entrada": {"x1": 0, "y1": 0, "x2": 400, "y2": 720},
        "zona_central": {"x1": 400, "y1": 0, "x2": 880, "y2": 720},
        "zona_salida": {"x1": 880, "y1": 0, "x2": 1280, "y2": 720}
    }

    # Mostrar zonas en video
    SHOW_ZONES = True

    # Opacidad de zonas (0.0 - 1.0)
    ZONE_OPACITY = 0.1


# ============================================================================
# CONFIGURACIÓN AVANZADA
# ============================================================================

class AdvancedConfig:
    """Configuración avanzada - no modificar a menos que sepas lo que haces"""

    # Usar GPU si está disponible
    USE_CUDA = True

    # Modelo de extractor de características
    FEATURE_EXTRACTOR_MODEL = 'osnet_x1_0'

    # Path del modelo (relativo a DATA_DIR)
    FEATURE_EXTRACTOR_PATH = 'osnet_x1_0_imagenet.pth'

    # Verbose output
    VERBOSE = False

    # Guardar video procesado
    SAVE_OUTPUT_VIDEO = False
    OUTPUT_VIDEO_PATH = "output_reid.mp4"
    OUTPUT_VIDEO_FPS = 30
