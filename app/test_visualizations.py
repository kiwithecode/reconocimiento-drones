"""
Script de prueba para verificar el sistema de visualizaciones
"""

import cv2
import numpy as np
import os
from visualization_handler import VisualizationHandler, VisualizationConfig

def create_test_image():
    """Crea una imagen de prueba con patrones diversos"""
    # Crear imagen de 128x256 con diferentes regiones
    img = np.zeros((256, 128, 3), dtype=np.uint8)

    # Región roja (superior)
    img[0:64, :] = [0, 0, 255]

    # Región azul (medio-superior)
    img[64:128, :] = [255, 0, 0]

    # Región verde (medio-inferior)
    img[128:192, :] = [0, 255, 0]

    # Región con patrón de rayas (inferior)
    for i in range(192, 256, 8):
        img[i:i+4, :] = [255, 255, 0]
        img[i+4:i+8, :] = [0, 255, 255]

    # Añadir ruido para texturas
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    return img

def test_visualization_system():
    """Prueba el sistema de visualizaciones"""

    print("="*70)
    print("🧪 TEST DEL SISTEMA DE VISUALIZACIONES")
    print("="*70)

    # Crear directorio de prueba
    test_dir = os.path.join(os.path.dirname(__file__), 'data', 'test_visualizaciones')
    os.makedirs(test_dir, exist_ok=True)

    print(f"\n📁 Directorio de prueba: {test_dir}")

    # Configurar sistema
    print("\n⚙️  Configurando sistema de visualizaciones...")
    viz_config = VisualizationConfig(test_dir)
    viz_config.SAVE_EVERY_N_FRAMES = 1  # Procesar todos los frames en test
    viz_config.MAX_SAVES_PER_PERSON = 3  # Solo 3 por persona en test
    viz_config.SAVE_ONLY_IDENTIFIED = False  # Guardar todo en test
    viz_config.MIN_CONFIDENCE_SAVE = 0.0  # Sin filtro de confianza en test

    viz_handler = VisualizationHandler(test_dir, config=viz_config)

    # Iniciar sistema
    print("▶️  Iniciando sistema...")
    viz_handler.start()

    # Crear imágenes de prueba
    print("\n📸 Generando imágenes de prueba...")

    test_cases = [
        {"name": "PersonaPrueba1", "confidence": 0.95},
        {"name": "PersonaPrueba2", "confidence": 0.85},
        {"name": "PersonaPrueba1", "confidence": 0.92},  # Repetida
    ]

    for i, test_case in enumerate(test_cases):
        # Crear imagen de prueba
        test_img = create_test_image()

        # Añadir a cola
        print(f"  [{i+1}/{len(test_cases)}] Procesando: {test_case['name']} (conf: {test_case['confidence']})")

        viz_handler.add_to_queue(
            crop=test_img,
            identity=test_case['name'],
            confidence=test_case['confidence'],
            track_id=i+1,
            frame_count=i+1
        )

    # Esperar a que se procesen todas
    print("\n⏳ Esperando procesamiento...")
    import time
    timeout = 30
    start_time = time.time()

    while viz_handler.queue.qsize() > 0 and (time.time() - start_time) < timeout:
        time.sleep(0.5)
        if viz_handler.queue.qsize() > 0:
            print(f"  Cola: {viz_handler.queue.qsize()} pendientes...", end='\r')

    print("\n")

    # Detener sistema
    print("⏹️  Deteniendo sistema...")
    viz_handler.stop()

    # Mostrar estadísticas
    stats = viz_handler.get_stats()

    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS DEL TEST")
    print("="*70)
    print(f"✅ Total procesadas: {stats['total_processed']}")
    print(f"⏭️  Total omitidas: {stats['total_skipped']}")
    print(f"📦 Cola final: {stats['queue_size']}")
    print(f"\n📊 Guardados por persona:")
    for person, count in stats['saves_per_person'].items():
        print(f"   - {person}: {count} visualización(es)")

    # Verificar archivos generados
    print(f"\n📁 Verificando archivos generados...")

    dirs_to_check = ['hsv', 'lbp', 'hog', 'original']
    total_files = 0

    for dir_name in dirs_to_check:
        dir_path = os.path.join(test_dir, dir_name)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            total_files += len(files)
            print(f"   - {dir_name}/: {len(files)} archivo(s)")

    print(f"\n✅ Total de archivos generados: {total_files}")

    # Test exitoso
    print("\n" + "="*70)
    if total_files > 0:
        print("✅ TEST EXITOSO")
        print(f"📂 Las visualizaciones están en: {test_dir}")
        print("\nPuedes revisar las imágenes generadas para verificar:")
        print("  - HSV: Análisis de colores")
        print("  - LBP: Análisis de texturas")
        print("  - HOG: Análisis de gradientes")
    else:
        print("⚠️  TEST COMPLETADO PERO NO SE GENERARON ARCHIVOS")
        print("Verifica que las dependencias estén instaladas correctamente")
    print("="*70)

def test_dependencies():
    """Verifica que todas las dependencias estén instaladas"""

    print("\n🔍 VERIFICANDO DEPENDENCIAS...")
    print("="*70)

    dependencies = {
        'OpenCV': 'cv2',
        'NumPy': 'numpy',
        'scikit-image': 'skimage'
    }

    all_ok = True

    for name, module_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✅ {name}: OK")
        except ImportError as e:
            print(f"❌ {name}: FALTA")
            print(f"   Instalar con: pip install {name.lower().replace(' ', '-')}")
            all_ok = False

    print("="*70)

    if all_ok:
        print("✅ Todas las dependencias están instaladas\n")
        return True
    else:
        print("❌ Faltan dependencias. Instálalas antes de continuar.\n")
        return False

if __name__ == "__main__":
    print("\n🚀 INICIANDO TESTS DE VISUALIZACIONES\n")

    # Verificar dependencias primero
    if test_dependencies():
        # Ejecutar test del sistema
        test_visualization_system()
    else:
        print("\n⚠️  Instala las dependencias faltantes y vuelve a ejecutar este script.")
        print("Comando sugerido:")
        print("  pip install opencv-python numpy scikit-image")

    print("\n✅ Tests finalizados\n")
