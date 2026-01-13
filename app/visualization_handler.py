"""
Sistema de Visualización en Segundo Plano para Re-ID
Genera visualizaciones de HSV, LBP y HOG de personas detectadas
"""

import os
import cv2
import numpy as np
from datetime import datetime
from threading import Thread, Lock
from queue import Queue, Empty
import logging
from skimage.feature import local_binary_pattern, hog
from skimage import exposure

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURACIÓN DE VISUALIZACIONES
# ============================================================================

class VisualizationConfig:
    """Configuración centralizada para visualizaciones"""

    # Activar/desactivar sistema completo
    ENABLE_VISUALIZATIONS = True

    # Qué visualizaciones generar
    ENABLE_HSV = True
    ENABLE_LBP = True
    ENABLE_HOG = True

    # Control de frecuencia
    SAVE_EVERY_N_FRAMES = 30  # Guardar cada 30 frames (~1 seg a 30fps)
    MAX_SAVES_PER_PERSON = 10  # Máximo de visualizaciones por persona

    # Filtros
    SAVE_ONLY_IDENTIFIED = True  # Solo guardar personas identificadas
    MIN_CONFIDENCE_SAVE = 0.6  # Mínima confianza para guardar

    # Tamaño de cola de procesamiento
    QUEUE_MAX_SIZE = 100

    # Formato de guardado
    IMAGE_FORMAT = "png"  # png o jpg
    IMAGE_QUALITY = 95  # Para JPG (0-100)

    # Parámetros LBP
    LBP_POINTS = 24  # Número de puntos circunferenciales
    LBP_RADIUS = 3  # Radio del círculo
    LBP_METHOD = 'uniform'  # 'uniform', 'default', 'ror', 'var'

    # Parámetros HOG
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)

    def __init__(self, output_base_dir):
        """Inicializa directorios de salida"""
        self.output_base_dir = output_base_dir

        # Crear estructura de carpetas
        self.dirs = {
            'hsv': os.path.join(output_base_dir, 'hsv'),
            'lbp': os.path.join(output_base_dir, 'lbp'),
            'hog': os.path.join(output_base_dir, 'hog'),
            'original': os.path.join(output_base_dir, 'original')
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        logger.info(f"✅ Directorios de visualización creados en: {output_base_dir}")


# ============================================================================
# PROCESADORES DE VISUALIZACIÓN
# ============================================================================

class VisualizationProcessors:
    """Procesadores de diferentes tipos de visualizaciones"""

    @staticmethod
    def process_hsv(crop):
        """
        Genera visualización HSV (Hue-Saturation-Value)
        Útil para análisis de colores de ropa
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Crear visualización combinada
        # Fila superior: H, S, V por separado
        # Fila inferior: Imagen HSV original, histograma de Hue

        # Normalizar canales para visualización
        h_vis = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
        s_vis = cv2.applyColorMap(s, cv2.COLORMAP_BONE)
        v_vis = cv2.applyColorMap(v, cv2.COLORMAP_BONE)

        # Redimensionar para visualización consistente
        h_vis = cv2.resize(h_vis, (128, 256))
        s_vis = cv2.resize(s_vis, (128, 256))
        v_vis = cv2.resize(v_vis, (128, 256))

        # Histograma de Hue (colores)
        hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
        hist_h = hist_h / hist_h.max() if hist_h.max() > 0 else hist_h

        # Crear imagen de histograma
        hist_img = np.zeros((256, 128, 3), dtype=np.uint8)
        bin_width = 128 / 180

        for i, val in enumerate(hist_h):
            color = cv2.cvtColor(np.uint8([[[i, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            height = int(val * 230)
            cv2.rectangle(hist_img,
                         (int(i * bin_width), 256 - height),
                         (int((i + 1) * bin_width), 256),
                         color.tolist(), -1)

        # Combinar visualizaciones
        top_row = np.hstack([h_vis, s_vis, v_vis])
        crop_resized = cv2.resize(crop, (128, 256))
        bottom_row = np.hstack([crop_resized, hist_img, np.zeros((256, 128, 3), dtype=np.uint8)])

        combined = np.vstack([top_row, bottom_row])

        # Añadir etiquetas
        cv2.putText(combined, "H", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "S", (138, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "V", (266, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Original", (10, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "Hue Hist", (138, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return combined

    @staticmethod
    def process_lbp(crop, n_points=24, radius=3, method='uniform'):
        """
        Genera visualización LBP (Local Binary Patterns)
        Útil para análisis de texturas y patrones de ropa
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Calcular LBP
        lbp = local_binary_pattern(gray, n_points, radius, method=method)

        # Normalizar para visualización
        lbp_normalized = ((lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255).astype(np.uint8)

        # Aplicar colormap para mejor visualización
        lbp_colored = cv2.applyColorMap(lbp_normalized, cv2.COLORMAP_JET)

        # Calcular histograma LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        # Crear visualización de histograma
        hist_height = 256
        hist_width = 384
        hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

        if hist.max() > 0:
            hist_normalized = hist / hist.max()
            bin_width = hist_width / len(hist)

            for i, val in enumerate(hist_normalized):
                height = int(val * (hist_height - 20))
                cv2.rectangle(hist_img,
                            (int(i * bin_width), hist_height - height),
                            (int((i + 1) * bin_width), hist_height),
                            (0, 255, 255), -1)

        # Redimensionar imágenes
        crop_resized = cv2.resize(crop, (128, 256))
        gray_resized = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (128, 256))
        lbp_resized = cv2.resize(lbp_colored, (128, 256))

        # Combinar visualizaciones
        top_row = np.hstack([crop_resized, gray_resized, lbp_resized])
        bottom_row = cv2.resize(hist_img, (384, 256))

        combined = np.vstack([top_row, bottom_row])

        # Añadir etiquetas
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "Grayscale", (138, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "LBP", (266, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined, "LBP Histogram", (10, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return combined

    @staticmethod
    def process_hog(crop, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Genera visualización HOG (Histogram of Oriented Gradients)
        Útil para análisis de forma y contornos
        """
        # Convertir a escala de grises
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Redimensionar a tamaño estándar para HOG
        gray_resized = cv2.resize(gray, (128, 256))

        # Calcular HOG con visualización
        try:
            fd, hog_image = hog(
                gray_resized,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                visualize=True,
                feature_vector=True
            )

            # Normalizar imagen HOG para mejor visualización
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            hog_image_rescaled = (hog_image_rescaled * 255).astype(np.uint8)

            # Aplicar colormap
            hog_colored = cv2.applyColorMap(hog_image_rescaled, cv2.COLORMAP_HOT)

            # Crear visualización de descriptores
            # Calcular histograma de orientaciones
            hist, bin_edges = np.histogram(fd, bins=50)
            hist_normalized = hist / hist.max() if hist.max() > 0 else hist

            # Crear imagen de histograma
            hist_img = np.zeros((256, 384, 3), dtype=np.uint8)
            bin_width = 384 / len(hist)

            for i, val in enumerate(hist_normalized):
                height = int(val * 230)
                cv2.rectangle(hist_img,
                            (int(i * bin_width), 256 - height),
                            (int((i + 1) * bin_width), 256),
                            (0, 255, 0), -1)

            # Combinar visualizaciones
            crop_resized = cv2.resize(crop, (128, 256))
            gray_colored = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (128, 256))
            hog_colored_resized = cv2.resize(hog_colored, (128, 256))

            top_row = np.hstack([crop_resized, gray_colored, hog_colored_resized])
            bottom_row = cv2.resize(hist_img, (384, 256))

            combined = np.vstack([top_row, bottom_row])

            # Añadir etiquetas
            cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Grayscale", (138, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "HOG", (266, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, f"HOG Features ({len(fd)})", (10, 286), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return combined

        except Exception as e:
            logger.error(f"Error en procesamiento HOG: {e}")
            # Retornar imagen de error
            error_img = np.zeros((512, 384, 3), dtype=np.uint8)
            cv2.putText(error_img, "Error HOG", (100, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return error_img


# ============================================================================
# MANEJADOR PRINCIPAL DE VISUALIZACIONES
# ============================================================================

class VisualizationHandler:
    """
    Manejador principal del sistema de visualizaciones en segundo plano
    """

    def __init__(self, output_base_dir, config=None):
        """
        Inicializa el manejador de visualizaciones

        Args:
            output_base_dir: Directorio base para guardar visualizaciones
            config: Objeto VisualizationConfig personalizado (opcional)
        """
        self.config = config or VisualizationConfig(output_base_dir)
        self.processors = VisualizationProcessors()

        # Cola de procesamiento
        self.queue = Queue(maxsize=self.config.QUEUE_MAX_SIZE)

        # Control de guardados por persona
        self.save_counts = {}  # {identity: count}
        self.lock = Lock()

        # Thread de procesamiento
        self.processing = False
        self.worker_thread = None

        # Estadísticas
        self.total_processed = 0
        self.total_skipped = 0

    def start(self):
        """Inicia el worker thread de procesamiento"""
        if not self.config.ENABLE_VISUALIZATIONS:
            logger.info("Sistema de visualizaciones desactivado")
            return

        self.processing = True
        self.worker_thread = Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("🎨 Sistema de visualizaciones iniciado")

    def stop(self):
        """Detiene el worker thread"""
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info(f"🎨 Sistema de visualizaciones detenido. Procesadas: {self.total_processed}, Omitidas: {self.total_skipped}")

    def add_to_queue(self, crop, identity, confidence, track_id, frame_count):
        """
        Añade una detección a la cola de procesamiento

        Args:
            crop: Imagen recortada de la persona (numpy array)
            identity: Identidad asignada (string)
            confidence: Confianza de la identificación (float)
            track_id: ID del track (int)
            frame_count: Número de frame actual (int)
        """
        if not self.config.ENABLE_VISUALIZATIONS:
            return

        # Filtros
        if self.config.SAVE_ONLY_IDENTIFIED and identity == "Desconocido":
            return

        if confidence < self.config.MIN_CONFIDENCE_SAVE:
            return

        # Control de frecuencia por frame
        if frame_count % self.config.SAVE_EVERY_N_FRAMES != 0:
            return

        # Control de límite por persona
        with self.lock:
            current_count = self.save_counts.get(identity, 0)
            if current_count >= self.config.MAX_SAVES_PER_PERSON:
                self.total_skipped += 1
                return

        # Añadir a la cola
        try:
            data = {
                'crop': crop.copy(),  # Copiar para evitar problemas de concurrencia
                'identity': identity,
                'confidence': confidence,
                'track_id': track_id,
                'timestamp': datetime.now(),
                'frame_count': frame_count
            }
            self.queue.put_nowait(data)
        except:
            # Cola llena, omitir
            self.total_skipped += 1

    def _process_queue(self):
        """Worker thread que procesa la cola de visualizaciones"""
        logger.info("Worker de visualizaciones iniciado")

        while self.processing:
            try:
                # Obtener item de la cola (timeout de 1 segundo)
                data = self.queue.get(timeout=1.0)

                # Procesar visualizaciones
                self._generate_visualizations(data)

                self.queue.task_done()

            except Empty:
                # No hay items, continuar
                continue
            except Exception as e:
                logger.error(f"Error en worker de visualizaciones: {e}")

        logger.info("Worker de visualizaciones finalizado")

    def _generate_visualizations(self, data):
        """
        Genera y guarda todas las visualizaciones para una detección

        Args:
            data: Diccionario con información de la detección
        """
        crop = data['crop']
        identity = data['identity']
        confidence = data['confidence']
        track_id = data['track_id']
        timestamp = data['timestamp']
        frame_count = data['frame_count']

        # Sanitizar nombre de identidad para usar como nombre de archivo
        safe_identity = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in identity)

        # Generar timestamp para nombre de archivo
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Incluir milisegundos

        # Nombre base del archivo
        base_filename = f"{safe_identity}_track{track_id}_frame{frame_count}_{ts_str}"

        try:
            # Guardar imagen original
            original_path = os.path.join(self.config.dirs['original'], f"{base_filename}.{self.config.IMAGE_FORMAT}")
            cv2.imwrite(original_path, crop)

            # Procesar HSV
            if self.config.ENABLE_HSV:
                hsv_vis = self.processors.process_hsv(crop)
                hsv_path = os.path.join(self.config.dirs['hsv'], f"{base_filename}_hsv.{self.config.IMAGE_FORMAT}")
                cv2.imwrite(hsv_path, hsv_vis)

            # Procesar LBP
            if self.config.ENABLE_LBP:
                lbp_vis = self.processors.process_lbp(
                    crop,
                    n_points=self.config.LBP_POINTS,
                    radius=self.config.LBP_RADIUS,
                    method=self.config.LBP_METHOD
                )
                lbp_path = os.path.join(self.config.dirs['lbp'], f"{base_filename}_lbp.{self.config.IMAGE_FORMAT}")
                cv2.imwrite(lbp_path, lbp_vis)

            # Procesar HOG
            if self.config.ENABLE_HOG:
                hog_vis = self.processors.process_hog(
                    crop,
                    orientations=self.config.HOG_ORIENTATIONS,
                    pixels_per_cell=self.config.HOG_PIXELS_PER_CELL,
                    cells_per_block=self.config.HOG_CELLS_PER_BLOCK
                )
                hog_path = os.path.join(self.config.dirs['hog'], f"{base_filename}_hog.{self.config.IMAGE_FORMAT}")
                cv2.imwrite(hog_path, hog_vis)

            # Actualizar contador
            with self.lock:
                self.save_counts[identity] = self.save_counts.get(identity, 0) + 1

            self.total_processed += 1

            # Log cada 10 visualizaciones
            if self.total_processed % 10 == 0:
                logger.info(f"🎨 Visualizaciones generadas: {self.total_processed} | Cola: {self.queue.qsize()}")

        except Exception as e:
            logger.error(f"Error generando visualizaciones para {identity}: {e}")

    def get_stats(self):
        """Retorna estadísticas del sistema de visualizaciones"""
        with self.lock:
            return {
                'total_processed': self.total_processed,
                'total_skipped': self.total_skipped,
                'queue_size': self.queue.qsize(),
                'saves_per_person': dict(self.save_counts)
            }
