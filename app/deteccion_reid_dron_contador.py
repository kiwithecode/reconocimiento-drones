import os
import cv2
from ultralytics import YOLO
import torch
import pickle
import numpy as np
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import time
from collections import defaultdict, deque
from datetime import datetime
import json
import csv
from config import DATA_DIR, BASE_EMBEDDINGS

# ============================================================================
# CONFIGURACIÓN OPTIMIZADA PARA DRON DJI FPV CON SISTEMA DE CONTEO
# ============================================================================

# Parámetros de procesamiento
FRAME_SKIP = 1
SCALE_PERCENT = 60
MIN_CONFIDENCE = 0.4

# Parámetros de Re-identificación
IDENTIFICATION_THRESHOLD = 0.55
REID_UPDATE_INTERVAL = 5

# Parámetros de Tracking
TRACK_PERSISTENCE_FRAMES = 15
IOU_THRESHOLD = 0.3
MAX_TRACK_AGE = 30

# Parámetros de Suavizado
SMOOTHING_WINDOW = 5
VOTE_WINDOW = 3

# Visualización
BBOX_THICKNESS = 3
FONT_SCALE = 0.8
CONFIDENCE_BAR_WIDTH = 100

# ============================================================================
# CONFIGURACIÓN DE CONTEO
# ============================================================================

# Sistema de conteo
ENABLE_COUNTING = True  # Activar sistema de conteo
COUNT_UNIQUE_ONLY = True  # Contar solo personas únicas (no repetir)

# Historial y estadísticas
SAVE_STATS_INTERVAL = 60  # Guardar estadísticas cada 60 segundos
EXPORT_FORMAT = "both"  # "csv", "json", "both"
STATS_OUTPUT_DIR = os.path.join(DATA_DIR, "estadisticas")

# Zonas de conteo (opcional)
ENABLE_ZONES = False  # Activar sistema de zonas
ZONES = {
    "zona_entrada": {"x1": 0, "y1": 0, "x2": 400, "y2": 720},
    "zona_central": {"x1": 400, "y1": 0, "x2": 880, "y2": 720},
    "zona_salida": {"x1": 880, "y1": 0, "x2": 1280, "y2": 720}
}

# ============================================================================
# CLASE PARA ESTADÍSTICAS Y CONTEO
# ============================================================================

class PeopleCounter:
    """Sistema de conteo y estadísticas de personas"""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or STATS_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        # Contadores actuales
        self.current_total = 0
        self.current_identified = 0
        self.current_unknown = 0

        # Contadores de personas únicas
        self.unique_tracks_seen = set()  # IDs de tracks únicos vistos
        self.unique_identified_people = set()  # Nombres únicos identificados
        self.unique_unknown_count = 0  # Personas desconocidas únicas

        # Contadores por zona (si están activadas)
        self.zone_counts = {zone: 0 for zone in ZONES.keys()} if ENABLE_ZONES else {}

        # Historial temporal
        self.history = []
        self.last_save_time = time.time()

        # Estadísticas de sesión
        self.session_start = datetime.now()
        self.total_frames_processed = 0
        self.peak_people = 0
        self.peak_time = None

        # Historial detallado de personas
        self.people_log = []

    def update(self, tracks):
        """Actualiza contadores con tracks actuales"""
        # Reiniciar contadores actuales
        self.current_total = 0
        self.current_identified = 0
        self.current_unknown = 0

        active_tracks = [t for t in tracks
                        if t.is_confirmed() and t.age < TRACK_PERSISTENCE_FRAMES]

        self.current_total = len(active_tracks)

        for track in active_tracks:
            # Agregar a tracks únicos vistos
            self.unique_tracks_seen.add(track.track_id)

            if track.current_identity != "Desconocido":
                self.current_identified += 1
                self.unique_identified_people.add(track.current_identity)
            else:
                self.current_unknown += 1

        # Actualizar pico de personas
        if self.current_total > self.peak_people:
            self.peak_people = self.current_total
            self.peak_time = datetime.now()

    def update_zones(self, tracks):
        """Actualiza contadores por zona"""
        if not ENABLE_ZONES:
            return

        # Reiniciar contadores de zonas
        for zone in self.zone_counts:
            self.zone_counts[zone] = 0

        active_tracks = [t for t in tracks
                        if t.is_confirmed() and t.age < TRACK_PERSISTENCE_FRAMES]

        for track in active_tracks:
            bbox = track.get_smoothed_bbox()
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2

            # Verificar en qué zona está
            for zone_name, zone_coords in ZONES.items():
                if (zone_coords["x1"] <= center_x <= zone_coords["x2"] and
                    zone_coords["y1"] <= center_y <= zone_coords["y2"]):
                    self.zone_counts[zone_name] += 1
                    break

    def log_detection(self, track_id, identity, confidence, timestamp):
        """Registra detección individual para análisis posterior"""
        self.people_log.append({
            "timestamp": timestamp.isoformat(),
            "track_id": int(track_id),
            "identity": identity,
            "confidence": float(confidence)
        })

    def add_to_history(self):
        """Agrega snapshot actual al historial"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "total": self.current_total,
            "identified": self.current_identified,
            "unknown": self.current_unknown,
            "zones": self.zone_counts.copy() if ENABLE_ZONES else {}
        })

    def get_unique_unknown_count(self):
        """Calcula personas desconocidas únicas"""
        # Personas desconocidas únicas = tracks únicos - personas identificadas únicas
        return max(0, len(self.unique_tracks_seen) - len(self.unique_identified_people))

    def get_stats_summary(self):
        """Retorna resumen de estadísticas"""
        session_duration = (datetime.now() - self.session_start).total_seconds()

        return {
            "session": {
                "start_time": self.session_start.isoformat(),
                "duration_seconds": session_duration,
                "frames_processed": self.total_frames_processed
            },
            "current": {
                "total_people": self.current_total,
                "identified": self.current_identified,
                "unknown": self.current_unknown,
                "zones": self.zone_counts if ENABLE_ZONES else {}
            },
            "unique": {
                "total_unique_tracks": len(self.unique_tracks_seen),
                "unique_identified_people": len(self.unique_identified_people),
                "identified_names": list(self.unique_identified_people),
                "unique_unknown": self.get_unique_unknown_count()
            },
            "peak": {
                "max_people_at_once": self.peak_people,
                "peak_time": self.peak_time.isoformat() if self.peak_time else None
            }
        }

    def convert_to_serializable(self, obj):
        """Convierte tipos numpy a tipos nativos de Python para JSON"""
        if isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_stats(self, format="both"):
        """Guarda estadísticas en archivo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats = self.get_stats_summary()

        if format in ["json", "both"]:
            json_path = os.path.join(self.output_dir, f"stats_{timestamp}.json")

            # Convertir datos a tipos serializables
            data_to_save = {
                "summary": self.convert_to_serializable(stats),
                "history": self.convert_to_serializable(self.history),
                "people_log": self.convert_to_serializable(self.people_log)
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            print(f"📊 Estadísticas guardadas: {json_path}")

        if format in ["csv", "both"]:
            csv_path = os.path.join(self.output_dir, f"stats_{timestamp}.csv")
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Métrica", "Valor"
                ])
                writer.writerow(["Total en área actual", stats["current"]["total_people"]])
                writer.writerow(["Identificados actual", stats["current"]["identified"]])
                writer.writerow(["Desconocidos actual", stats["current"]["unknown"]])
                writer.writerow(["Total personas únicas vistas", stats["unique"]["total_unique_tracks"]])
                writer.writerow(["Personas identificadas únicas", stats["unique"]["unique_identified_people"]])
                writer.writerow(["Personas desconocidas únicas", stats["unique"]["unique_unknown"]])
                writer.writerow(["Pico de personas simultáneas", stats["peak"]["max_people_at_once"]])
                writer.writerow(["Duración sesión (seg)", int(stats["session"]["duration_seconds"])])
            print(f"📊 CSV guardado: {csv_path}")

    def should_save(self):
        """Verifica si es momento de guardar estadísticas"""
        return (time.time() - self.last_save_time) >= SAVE_STATS_INTERVAL

    def mark_saved(self):
        """Marca timestamp de última guardada"""
        self.last_save_time = time.time()

# ============================================================================
# CLASE PARA TRACKING DE PERSONAS
# ============================================================================

class PersonTrack:
    """Representa un track de una persona a través de múltiples frames"""
    _next_id = 1

    def __init__(self, bbox, frame_id):
        self.track_id = PersonTrack._next_id
        PersonTrack._next_id += 1
        self.bbox = bbox
        self.bbox_history = deque([bbox], maxlen=SMOOTHING_WINDOW)
        self.identities = deque(maxlen=VOTE_WINDOW)
        self.confidence_scores = deque(maxlen=VOTE_WINDOW)
        self.current_identity = "Desconocido"
        self.current_confidence = 0.0
        self.age = 0
        self.total_frames = 0
        self.last_seen_frame = frame_id
        self.last_reid_frame = frame_id
        self.color = self._generate_color()
        self.first_detection_time = datetime.now()

    def _generate_color(self):
        np.random.seed(self.track_id)
        return tuple(map(int, np.random.randint(100, 255, 3)))

    def update(self, bbox, frame_id):
        self.bbox = bbox
        self.bbox_history.append(bbox)
        self.age = 0
        self.total_frames += 1
        self.last_seen_frame = frame_id

    def update_identity(self, identity, confidence, frame_id):
        self.identities.append(identity)
        self.confidence_scores.append(confidence)
        self.last_reid_frame = frame_id
        if len(self.identities) > 0:
            identity_counts = defaultdict(int)
            for ident in self.identities:
                identity_counts[ident] += 1
            self.current_identity = max(identity_counts.items(), key=lambda x: x[1])[0]
            self.current_confidence = float(np.mean(list(self.confidence_scores)))

    def get_smoothed_bbox(self):
        if len(self.bbox_history) == 0:
            return self.bbox
        bboxes = np.array(list(self.bbox_history))
        smoothed = np.mean(bboxes, axis=0).astype(int)
        return tuple(smoothed)

    def increment_age(self):
        self.age += 1

    def is_confirmed(self):
        return self.total_frames >= 3

    def is_alive(self):
        return self.age < MAX_TRACK_AGE

# ============================================================================
# FUNCIONES DE TRACKING (mismo código anterior)
# ============================================================================

def calculate_iou(bbox1, bbox2):
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def match_detections_to_tracks(detections, tracks):
    if len(tracks) == 0:
        return [], list(range(len(detections))), []
    if len(detections) == 0:
        return [], [], list(range(len(tracks)))

    iou_matrix = np.zeros((len(detections), len(tracks)))
    for d_idx, det_bbox in enumerate(detections):
        for t_idx, track in enumerate(tracks):
            iou_matrix[d_idx, t_idx] = calculate_iou(det_bbox, track.bbox)

    matched = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(range(len(tracks)))

    while True:
        if len(unmatched_detections) == 0 or len(unmatched_tracks) == 0:
            break
        max_iou = 0
        max_det_idx = -1
        max_track_idx = -1
        for d_idx in unmatched_detections:
            for t_idx in unmatched_tracks:
                if iou_matrix[d_idx, t_idx] > max_iou:
                    max_iou = iou_matrix[d_idx, t_idx]
                    max_det_idx = d_idx
                    max_track_idx = t_idx
        if max_iou >= IOU_THRESHOLD:
            matched.append((max_det_idx, max_track_idx))
            unmatched_detections.remove(max_det_idx)
            unmatched_tracks.remove(max_track_idx)
        else:
            break

    return matched, unmatched_detections, unmatched_tracks

def identificar(embedding, base, threshold=IDENTIFICATION_THRESHOLD):
    if not base:
        return "Desconocido", 0.0
    emb_np = embedding.cpu().numpy()
    base_embs = np.array([p["embedding"] for p in base])
    similarities = cosine_similarity([emb_np], base_embs)[0]
    max_idx = np.argmax(similarities)
    mejor_score = similarities[max_idx]
    if mejor_score >= threshold:
        return base[max_idx]["nombre"], mejor_score
    return "Desconocido", mejor_score

# ============================================================================
# VISUALIZACIÓN MEJORADA CON CONTADORES
# ============================================================================

def draw_track_info(frame, track, show_confidence_bar=True):
    """Dibuja información del track"""
    bbox = track.get_smoothed_bbox()
    x1, y1, x2, y2 = bbox

    if track.current_identity != "Desconocido":
        color = (0, 255, 0)
        label = f"{track.current_identity}"
    else:
        color = (0, 165, 255)
        label = f"ID:{track.track_id} - Desconocido"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

    conf_text = f" ({track.current_confidence:.2f})"
    full_label = label + conf_text

    (text_width, text_height), baseline = cv2.getTextSize(
        full_label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 2
    )

    cv2.rectangle(frame,
                  (x1, y1 - text_height - 10),
                  (x1 + text_width, y1),
                  color, -1)

    cv2.putText(frame, full_label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 2)

    if show_confidence_bar and track.current_identity != "Desconocido":
        bar_height = 8
        bar_y = y2 + 5
        cv2.rectangle(frame, (x1, bar_y),
                      (x1 + CONFIDENCE_BAR_WIDTH, bar_y + bar_height),
                      (100, 100, 100), -1)
        conf_width = int(CONFIDENCE_BAR_WIDTH * track.current_confidence)
        conf_color = (0, int(255 * track.current_confidence),
                      int(255 * (1 - track.current_confidence)))
        cv2.rectangle(frame, (x1, bar_y),
                      (x1 + conf_width, bar_y + bar_height),
                      conf_color, -1)

def draw_zones(frame):
    """Dibuja zonas de conteo en el frame"""
    if not ENABLE_ZONES:
        return

    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]

    for i, (zone_name, coords) in enumerate(ZONES.items()):
        color = colors[i % len(colors)]
        # Dibujar rectángulo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (coords["x1"], coords["y1"]),
                     (coords["x2"], coords["y2"]),
                     color, -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)

        # Dibujar borde
        cv2.rectangle(frame,
                     (coords["x1"], coords["y1"]),
                     (coords["x2"], coords["y2"]),
                     color, 2)

        # Etiqueta de zona
        cv2.putText(frame, zone_name,
                   (coords["x1"] + 10, coords["y1"] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def draw_counting_panel(frame, counter, fps):
    """Dibuja panel de conteo y estadísticas"""
    panel_width = 350
    panel_height = 450 if ENABLE_ZONES else 350
    panel_x = frame.shape[1] - panel_width - 10
    panel_y = 10

    # Fondo del panel
    overlay = frame.copy()
    cv2.rectangle(overlay,
                 (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Borde del panel
    cv2.rectangle(frame,
                 (panel_x, panel_y),
                 (panel_x + panel_width, panel_y + panel_height),
                 (255, 255, 255), 2)

    # Título
    title_y = panel_y + 30
    cv2.putText(frame, "CONTEO DE PERSONAS",
               (panel_x + 10, title_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Línea separadora
    cv2.line(frame,
            (panel_x + 10, title_y + 10),
            (panel_x + panel_width - 10, title_y + 10),
            (255, 255, 255), 1)

    # Información actual
    y_offset = title_y + 40

    stats = [
        ("EN AREA ACTUAL:", ""),
        (f"  Total:", f"{counter.current_total}", (255, 255, 255)),
        (f"  Identificados:", f"{counter.current_identified}", (0, 255, 0)),
        (f"  Desconocidos:", f"{counter.current_unknown}", (0, 165, 255)),
        ("", ""),
        ("PERSONAS UNICAS:", ""),
        (f"  Total vistas:", f"{len(counter.unique_tracks_seen)}", (255, 255, 255)),
        (f"  Identificadas:", f"{len(counter.unique_identified_people)}", (0, 255, 0)),
        (f"  Desconocidas:", f"{counter.get_unique_unknown_count()}", (0, 165, 255)),
        ("", ""),
        ("ESTADISTICAS:", ""),
        (f"  FPS:", f"{fps:.1f}", (255, 255, 255)),
        (f"  Pico personas:", f"{counter.peak_people}", (255, 200, 0)),
    ]

    for stat in stats:
        if len(stat) == 2:  # Línea vacía o título
            label, value = stat
            color = (255, 255, 255)
        else:
            label, value, color = stat

        # Etiqueta
        cv2.putText(frame, label,
                   (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Valor (si existe)
        if value:
            cv2.putText(frame, str(value),
                       (panel_x + panel_width - 80, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        y_offset += 25

    # Conteo por zonas (si está activo)
    if ENABLE_ZONES:
        y_offset += 10
        cv2.putText(frame, "CONTEO POR ZONAS:",
                   (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

        for zone_name, count in counter.zone_counts.items():
            cv2.putText(frame, f"  {zone_name}:",
                       (panel_x + 15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, str(count),
                       (panel_x + panel_width - 80, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            y_offset += 25

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🚁 SISTEMA DE RE-IDENTIFICACIÓN CON CONTEO - DRON DJI FPV")
    print("="*70)

    cv2.namedWindow('ReID + Contador', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ReID + Contador', 1280, 720)

    # Cargar YOLO
    print("\n📦 Cargando YOLO...")
    model = YOLO("yolov8n.pt")
    model.conf = MIN_CONFIDENCE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"✅ YOLO en {device}")

    # Cargar extractor
    print("\n🧠 Cargando OSNet...")
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=os.path.join(DATA_DIR, 'osnet_x1_0_imagenet.pth'),
        device=device
    )
    print("✅ Extractor cargado")

    # Cargar base
    print(f"\n📚 Cargando base de datos...")
    with open(BASE_EMBEDDINGS, "rb") as f:
        base = pickle.load(f)
    print(f"✅ Base: {len(base)} persona(s)")

    # Inicializar contador
    counter = PeopleCounter()
    print(f"📊 Sistema de conteo activado")
    print(f"📁 Estadísticas se guardarán en: {counter.output_dir}")

    # Conectar stream
    print("\n🎥 Conectando a stream RTMP...")
    rtmp_url = "rtmp://localhost:1935/live/stream"
    cap = cv2.VideoCapture(rtmp_url)

    if not cap.isOpened():
        print("⚠️ No se pudo conectar a RTMP. Probando con webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se pudo conectar")
            return

    print("✅ Stream conectado")
    print("\n" + "="*70)
    print("🚀 INICIANDO DETECCIÓN Y CONTEO")
    print("="*70)
    print("⌨️  Controles:")
    print("   - 'q': Salir (guarda estadísticas)")
    print("   - 's': Guardar estadísticas ahora")
    print("   - 'r': Reset contadores")
    print()

    # Variables
    tracks = []
    frame_count = 0
    fps = 0
    fps_counter = 0
    fps_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fps_counter += 1
        counter.total_frames_processed += 1

        # Procesamiento
        original_frame = frame.copy()

        if SCALE_PERCENT != 100:
            width = int(frame.shape[1] * SCALE_PERCENT / 100)
            height = int(frame.shape[0] * SCALE_PERCENT / 100)
            frame_small = cv2.resize(frame, (width, height))
            scale_factor = 100 / SCALE_PERCENT
        else:
            frame_small = frame
            scale_factor = 1.0

        process_this_frame = (frame_count % FRAME_SKIP == 0)

        if process_this_frame:
            results = model(frame_small, verbose=False, imgsz=640)
            detections = []
            detection_crops = []

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        conf = float(box.conf[0])
                        if conf >= MIN_CONFIDENCE:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            if scale_factor != 1.0:
                                x1, y1 = int(x1 * scale_factor), int(y1 * scale_factor)
                                x2, y2 = int(x2 * scale_factor), int(y2 * scale_factor)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2 = min(original_frame.shape[1], x2)
                            y2 = min(original_frame.shape[0], y2)
                            if x2 > x1 and y2 > y1:
                                detections.append((x1, y1, x2, y2))
                                crop = original_frame[y1:y2, x1:x2]
                                detection_crops.append(crop)

            matched, unmatched_dets, unmatched_tracks = match_detections_to_tracks(
                detections, tracks
            )

            for det_idx, track_idx in matched:
                tracks[track_idx].update(detections[det_idx], frame_count)

            for det_idx in unmatched_dets:
                new_track = PersonTrack(detections[det_idx], frame_count)
                tracks.append(new_track)

            for track_idx in unmatched_tracks:
                tracks[track_idx].increment_age()

            # Re-ID
            for i, track in enumerate(tracks):
                should_reid = (
                    track.is_confirmed() and
                    track.age == 0 and
                    (frame_count - track.last_reid_frame) >= REID_UPDATE_INTERVAL
                )

                if should_reid:
                    crop_idx = None
                    for j, (det_idx, track_idx) in enumerate(matched):
                        if track_idx == i:
                            crop_idx = det_idx
                            break

                    if crop_idx is not None and crop_idx < len(detection_crops):
                        crop = detection_crops[crop_idx]
                        if crop.size > 0:
                            try:
                                crop_resized = cv2.resize(crop, (128, 256))
                                emb = extractor(crop_resized)[0]
                                identity, confidence = identificar(emb, base)
                                track.update_identity(identity, confidence, frame_count)

                                # Log detección
                                counter.log_detection(
                                    track.track_id,
                                    identity,
                                    confidence,
                                    datetime.now()
                                )
                            except Exception as e:
                                pass

            tracks = [t for t in tracks if t.is_alive()]

        # Actualizar contadores
        counter.update(tracks)
        if ENABLE_ZONES:
            counter.update_zones(tracks)

        # Agregar a historial cada segundo
        if frame_count % 30 == 0:  # Asumiendo ~30 FPS
            counter.add_to_history()

        # Guardar estadísticas periódicamente
        if counter.should_save():
            counter.save_stats(EXPORT_FORMAT)
            counter.mark_saved()

        # Dibujar zonas
        draw_zones(original_frame)

        # Visualización de tracks
        for track in tracks:
            if track.is_confirmed() and track.age < TRACK_PERSISTENCE_FRAMES:
                draw_track_info(original_frame, track, show_confidence_bar=True)

        # FPS
        current_time = time.time()
        if current_time - fps_start_time >= 1.0:
            fps = fps_counter / (current_time - fps_start_time)
            fps_counter = 0
            fps_start_time = current_time

        # Panel de conteo
        draw_counting_panel(original_frame, counter, fps)

        cv2.imshow("ReID + Contador", original_frame)

        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n💾 Guardando estadísticas finales...")
            counter.save_stats(EXPORT_FORMAT)
            break
        elif key == ord('s'):
            print("\n💾 Guardando estadísticas...")
            counter.save_stats(EXPORT_FORMAT)
        elif key == ord('r'):
            print("\n🔄 Reseteando contadores...")
            counter = PeopleCounter()

    # Resumen final
    print("\n" + "="*70)
    print("📊 RESUMEN DE SESIÓN")
    print("="*70)
    stats = counter.get_stats_summary()
    print(f"⏱️  Duración: {stats['session']['duration_seconds']:.1f} segundos")
    print(f"👥 Total personas únicas detectadas: {stats['unique']['total_unique_tracks']}")
    print(f"✅ Personas identificadas: {stats['unique']['unique_identified_people']}")
    print(f"❓ Personas desconocidas: {stats['unique']['unique_unknown']}")
    print(f"📈 Pico de personas simultáneas: {stats['peak']['max_people_at_once']}")
    if stats['unique']['identified_names']:
        print(f"📝 Nombres identificados: {', '.join(stats['unique']['identified_names'])}")
    print("="*70)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
