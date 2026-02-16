"""
Generador mejorado de embeddings para re-identificación

Mejoras implementadas:
- Data augmentation (flip horizontal, ajustes de brillo)
- Filtrado de outliers usando distancia de Mahalanobis
- Promedio ponderado por calidad
- Validación de consistencia de embeddings
- Normalización L2
- Visualización de distribución
- Reporte detallado de calidad
"""

import os
import cv2
import torch
import pickle
import numpy as np
from torchreid.utils import FeatureExtractor
from scipy.spatial.distance import cdist, mahalanobis
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from config import PERSONAS_BASE, BASE_EMBEDDINGS, MODEL_PATH, DATA_DIR

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Data Augmentation
USE_AUGMENTATION = True
AUGMENTATION_FLIP = True  # Flip horizontal
AUGMENTATION_BRIGHTNESS = True  # Ajustes de brillo

# Filtrado de outliers
FILTER_OUTLIERS = True
OUTLIER_THRESHOLD = 2.5  # Desviaciones estándar
MIN_IMAGES_FOR_FILTERING = 5  # Mínimo de imágenes para filtrar

# Promedio ponderado
USE_WEIGHTED_AVERAGE = True
QUALITY_WEIGHT_POWER = 2  # Potencia para pesos de calidad

# Normalización
NORMALIZE_EMBEDDINGS = True

# ============================================================================
# FUNCIONES DE AUGMENTATION
# ============================================================================

def augment_image(image):
    """
    Genera versiones aumentadas de una imagen

    Returns:
        list: Lista de imágenes aumentadas
    """
    augmented = [image]  # Siempre incluir original

    if not USE_AUGMENTATION:
        return augmented

    # Flip horizontal
    if AUGMENTATION_FLIP:
        flipped = cv2.flip(image, 1)
        augmented.append(flipped)

    # Ajustes de brillo
    if AUGMENTATION_BRIGHTNESS:
        # Más brillante
        bright = cv2.convertScaleAbs(image, alpha=1.2, beta=20)
        augmented.append(bright)

        # Más oscuro
        dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
        augmented.append(dark)

    return augmented

# ============================================================================
# FUNCIONES DE CALIDAD
# ============================================================================

def calculate_image_quality(image):
    """
    Calcula métricas de calidad de una imagen

    Returns:
        float: Score de calidad (0-100)
    """
    # Nitidez (Laplacian variance)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    # Brillo
    brightness = np.mean(gray)

    # Score de calidad
    quality = 50  # Base

    # Bonus por nitidez
    if sharpness > 100:
        quality += min(30, (sharpness - 100) / 20)

    # Penalización por brillo extremo
    if brightness < 50 or brightness > 200:
        quality -= 20

    # Bonus por brillo óptimo
    if 80 <= brightness <= 160:
        quality += 20

    return max(0, min(100, quality))

# ============================================================================
# FUNCIONES DE FILTRADO
# ============================================================================

def filter_outlier_embeddings(embeddings, threshold=OUTLIER_THRESHOLD):
    """
    Filtra embeddings outliers usando distancia de Mahalanobis

    Args:
        embeddings: numpy array de embeddings (n_samples, n_features)
        threshold: umbral en desviaciones estándar

    Returns:
        tuple: (embeddings_filtrados, indices_válidos, outlier_scores)
    """
    if len(embeddings) < MIN_IMAGES_FOR_FILTERING:
        # No suficientes imágenes para filtrar confiablemente
        return embeddings, list(range(len(embeddings))), np.zeros(len(embeddings))

    # Calcular centroide y covarianza
    mean_emb = np.mean(embeddings, axis=0)
    cov_matrix = np.cov(embeddings.T)

    # Evitar matriz singular
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

    # Calcular distancia de Mahalanobis para cada embedding
    try:
        cov_inv = np.linalg.inv(cov_matrix)
        mahal_distances = np.array([
            mahalanobis(emb, mean_emb, cov_inv)
            for emb in embeddings
        ])
    except np.linalg.LinAlgError:
        # Si falla, usar distancia euclidiana normalizada
        distances = cdist(embeddings, [mean_emb], metric='euclidean').flatten()
        mahal_distances = (distances - np.mean(distances)) / (np.std(distances) + 1e-10)

    # Convertir a z-scores
    z_scores = (mahal_distances - np.mean(mahal_distances)) / (np.std(mahal_distances) + 1e-10)

    # Filtrar outliers
    valid_indices = np.where(np.abs(z_scores) <= threshold)[0]

    filtered_embeddings = embeddings[valid_indices]

    return filtered_embeddings, valid_indices.tolist(), z_scores

def calculate_embedding_consistency(embeddings):
    """
    Calcula la consistencia de un conjunto de embeddings

    Returns:
        float: Score de consistencia (0-100)
    """
    if len(embeddings) < 2:
        return 100.0

    # Calcular distancias entre todos los pares
    distances = cdist(embeddings, embeddings, metric='cosine')

    # Distancia promedio (excluir diagonal)
    mask = ~np.eye(distances.shape[0], dtype=bool)
    avg_distance = np.mean(distances[mask])

    # Convertir a score de similitud
    avg_similarity = 1 - avg_distance

    # Normalizar a 0-100
    consistency_score = avg_similarity * 100

    return max(0, min(100, consistency_score))

# ============================================================================
# GENERACIÓN DE EMBEDDINGS
# ============================================================================

def generate_embeddings_for_person(person_dir, person_name, extractor):
    """
    Genera embeddings para una persona con todas las mejoras

    Returns:
        dict: Información de embeddings y estadísticas
    """
    print(f"\n{'='*70}")
    print(f"Procesando: {person_name}")
    print(f"{'='*70}")

    embeddings = []
    quality_scores = []
    image_names = []

    # Procesar cada imagen
    image_files = [f for f in os.listdir(person_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("⚠️ No se encontraron imágenes")
        return None

    print(f"📸 Imágenes encontradas: {len(image_files)}")

    for img_file in image_files:
        img_path = os.path.join(person_dir, img_file)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ No se pudo leer: {img_file}")
            continue

        # Calcular calidad
        quality = calculate_image_quality(image)

        # Redimensionar
        image_resized = cv2.resize(image, (128, 256))

        # Data augmentation
        augmented_images = augment_image(image_resized)

        # Extraer embeddings de todas las versiones
        for aug_img in augmented_images:
            emb = extractor(aug_img)[0].cpu().numpy()
            embeddings.append(emb)
            quality_scores.append(quality)
            image_names.append(img_file)

    if not embeddings:
        print("❌ No se generaron embeddings")
        return None

    embeddings = np.array(embeddings)
    quality_scores = np.array(quality_scores)

    print(f"🧠 Embeddings generados: {len(embeddings)}")
    if USE_AUGMENTATION:
        print(f"   (incluyendo augmentation)")

    # Filtrar outliers
    if FILTER_OUTLIERS:
        filtered_embs, valid_indices, outlier_scores = filter_outlier_embeddings(embeddings)

        n_removed = len(embeddings) - len(filtered_embs)
        if n_removed > 0:
            print(f"🔍 Outliers removidos: {n_removed}/{len(embeddings)}")

        embeddings = filtered_embs
        quality_scores = quality_scores[valid_indices]
    else:
        outlier_scores = np.zeros(len(embeddings))

    # Calcular consistencia
    consistency = calculate_embedding_consistency(embeddings)
    print(f"📊 Consistencia: {consistency:.1f}%")

    # Promedio ponderado o simple
    if USE_WEIGHTED_AVERAGE and len(embeddings) > 1:
        # Normalizar quality scores a pesos
        weights = np.power(quality_scores / 100.0, QUALITY_WEIGHT_POWER)
        weights = weights / np.sum(weights)

        # Promedio ponderado
        final_embedding = np.average(embeddings, axis=0, weights=weights)
        print(f"⚖️  Promedio ponderado por calidad")
    else:
        # Promedio simple
        final_embedding = np.mean(embeddings, axis=0)
        print(f"📊 Promedio simple")

    # Normalización L2
    if NORMALIZE_EMBEDDINGS:
        norm = np.linalg.norm(final_embedding)
        if norm > 0:
            final_embedding = final_embedding / norm
        print(f"🔧 Embedding normalizado (L2)")

    # Calcular calidad promedio
    avg_quality = np.mean(quality_scores)

    print(f"✅ Embedding final generado")
    print(f"   Calidad promedio: {avg_quality:.1f}%")
    print(f"   Consistencia: {consistency:.1f}%")

    # Advertencias
    if consistency < 60:
        print(f"⚠️  ADVERTENCIA: Baja consistencia - puede afectar precisión")
    if avg_quality < 60:
        print(f"⚠️  ADVERTENCIA: Baja calidad promedio - recapturar recomendado")
    if len(image_files) < 8:
        print(f"⚠️  ADVERTENCIA: Pocas imágenes ({len(image_files)}) - capturar más recomendado")

    return {
        "embedding": final_embedding,
        "n_images": len(image_files),
        "n_embeddings": len(embeddings),
        "avg_quality": avg_quality,
        "consistency": consistency,
        "outliers_removed": len(image_files) - len(valid_indices) if FILTER_OUTLIERS else 0
    }

# ============================================================================
# EVALUACIÓN - MATRIZ DE CONFUSIÓN
# ============================================================================

def evaluar_modelo(base_embeddings, personas_base_dir, extractor, threshold=0.70):
    """
    Evalúa el modelo con matriz de confusión

    Compara cada imagen individual contra la base de datos
    para medir precisión, recall y F1-score.
    """
    print("\n" + "="*70)
    print("📊 EVALUANDO MODELO - MATRIZ DE CONFUSIÓN")
    print("="*70)
    print(f"   Umbral de identificación: {threshold}")

    etiquetas_reales = []
    predicciones = []
    confidencias = []

    # Por cada persona en la base
    for persona_folder in os.listdir(personas_base_dir):
        persona_dir = os.path.join(personas_base_dir, persona_folder)
        if not os.path.isdir(persona_dir):
            continue

        # Nombre real (ground truth)
        partes = persona_folder.split("_")
        if len(partes) >= 3:
            nombre_real = " ".join(partes[:-1]).replace("_", " ").title()
        else:
            nombre_real = persona_folder.replace("_", " ").title()

        # Por cada imagen de la persona
        for img_file in os.listdir(persona_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(persona_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            # Generar embedding de la imagen
            image_resized = cv2.resize(image, (128, 256))
            emb = extractor(image_resized)[0].cpu().numpy()

            # Normalizar embedding
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm

            # Comparar con base de datos
            base_embs = np.array([p["embedding"] for p in base_embeddings])
            similarities = cosine_similarity([emb], base_embs)[0]
            max_idx = np.argmax(similarities)
            mejor_score = similarities[max_idx]

            # Predicción
            if mejor_score >= threshold:
                prediccion = base_embeddings[max_idx]["nombre"]
            else:
                prediccion = "Desconocido"

            etiquetas_reales.append(nombre_real)
            predicciones.append(prediccion)
            confidencias.append(mejor_score)

    # Obtener todas las etiquetas únicas
    todas_etiquetas = sorted(list(set(etiquetas_reales + predicciones)))

    # Generar matriz de confusión
    cm = confusion_matrix(etiquetas_reales, predicciones, labels=todas_etiquetas)

    # Mostrar matriz de confusión
    print("\n📋 MATRIZ DE CONFUSIÓN:")
    print("-"*70)

    # Header
    header = "Real \\ Pred".ljust(20)
    for label in todas_etiquetas:
        header += label[:10].center(12)
    print(header)
    print("-"*70)

    # Filas
    for i, label in enumerate(todas_etiquetas):
        row = label[:18].ljust(20)
        for j in range(len(todas_etiquetas)):
            row += str(cm[i][j]).center(12)
        print(row)

    print("-"*70)

    # Reporte de clasificación
    print("\n📈 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(etiquetas_reales, predicciones, labels=todas_etiquetas, zero_division=0))

    # Métricas adicionales
    total = len(etiquetas_reales)
    correctos = sum(1 for r, p in zip(etiquetas_reales, predicciones) if r == p)
    accuracy = correctos / total * 100 if total > 0 else 0

    print(f"\n📊 RESUMEN:")
    print(f"   Total de imágenes evaluadas: {total}")
    print(f"   Predicciones correctas: {correctos}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Confianza promedio: {np.mean(confidencias)*100:.2f}%")

    # Guardar resultados en CSV
    results_path = os.path.join(DATA_DIR, "matriz_confusion_resultados.csv")
    with open(results_path, 'w') as f:
        f.write("Etiqueta Real,Prediccion,Confianza,Correcto\n")
        for real, pred, conf in zip(etiquetas_reales, predicciones, confidencias):
            correcto = "SI" if real == pred else "NO"
            f.write(f"{real},{pred},{conf:.4f},{correcto}\n")

    print(f"\n💾 Resultados CSV guardados en: {results_path}")

    # =========================================================================
    # VISUALIZACIÓN GRÁFICA DE LA MATRIZ DE CONFUSIÓN
    # =========================================================================
    print("\n🎨 Generando visualización gráfica...")

    # Crear figura con 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Matriz de Confusión (valores absolutos) ---
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=todas_etiquetas,
                yticklabels=todas_etiquetas,
                ax=axes[0],
                cbar_kws={'label': 'Cantidad'})
    axes[0].set_title('Matriz de Confusión\n(Valores Absolutos)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicción', fontsize=12)
    axes[0].set_ylabel('Etiqueta Real', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # --- Matriz de Confusión (porcentajes normalizados) ---
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Reemplazar NaN por 0

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=todas_etiquetas,
                yticklabels=todas_etiquetas,
                ax=axes[1],
                cbar_kws={'label': 'Porcentaje'})
    axes[1].set_title('Matriz de Confusión\n(Normalizada por Fila)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicción', fontsize=12)
    axes[1].set_ylabel('Etiqueta Real', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    # Título general
    fig.suptitle(f'Evaluación del Modelo de Re-Identificación\nAccuracy: {accuracy:.2f}% | Umbral: {threshold}',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Guardar imagen
    img_path = os.path.join(DATA_DIR, "matriz_confusion_visual.png")
    plt.savefig(img_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📊 Imagen guardada en: {img_path}")

    # Mostrar gráfico
    plt.show()

    # --- Gráfico adicional: Métricas por persona ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Calcular precision y recall por persona
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        etiquetas_reales, predicciones, labels=todas_etiquetas, zero_division=0
    )

    # Gráfico de barras - Precision y Recall
    x = np.arange(len(todas_etiquetas))
    width = 0.35

    bars1 = axes2[0].bar(x - width/2, precision, width, label='Precisión', color='steelblue')
    bars2 = axes2[0].bar(x + width/2, recall, width, label='Recall', color='coral')

    axes2[0].set_xlabel('Persona', fontsize=12)
    axes2[0].set_ylabel('Score', fontsize=12)
    axes2[0].set_title('Precisión y Recall por Persona', fontsize=14, fontweight='bold')
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels([label[:15] for label in todas_etiquetas], rotation=45, ha='right')
    axes2[0].legend()
    axes2[0].set_ylim(0, 1.1)
    axes2[0].axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Umbral 70%')

    # Agregar valores encima de las barras
    for bar in bars1:
        height = bar.get_height()
        axes2[0].annotate(f'{height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        axes2[0].annotate(f'{height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)

    # Gráfico de F1-Score
    colors = ['green' if f >= 0.7 else 'orange' if f >= 0.5 else 'red' for f in f1]
    bars3 = axes2[1].bar(todas_etiquetas, f1, color=colors)
    axes2[1].set_xlabel('Persona', fontsize=12)
    axes2[1].set_ylabel('F1-Score', fontsize=12)
    axes2[1].set_title('F1-Score por Persona', fontsize=14, fontweight='bold')
    axes2[1].tick_params(axis='x', rotation=45)
    axes2[1].set_ylim(0, 1.1)
    axes2[1].axhline(y=0.7, color='red', linestyle='--', alpha=0.5)

    # Agregar valores
    for bar, val in zip(bars3, f1):
        axes2[1].annotate(f'{val:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, val),
                          xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Guardar segunda imagen
    img_path2 = os.path.join(DATA_DIR, "metricas_por_persona.png")
    plt.savefig(img_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📈 Métricas guardadas en: {img_path2}")

    plt.show()

    return cm, accuracy

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("🧠 GENERADOR MEJORADO DE EMBEDDINGS")
    print("="*70)

    print("\nConfiguración:")
    print(f"  Data Augmentation: {'✅' if USE_AUGMENTATION else '❌'}")
    print(f"  Filtrado de Outliers: {'✅' if FILTER_OUTLIERS else '❌'}")
    print(f"  Promedio Ponderado: {'✅' if USE_WEIGHTED_AVERAGE else '❌'}")
    print(f"  Normalización L2: {'✅' if NORMALIZE_EMBEDDINGS else '❌'}")

    # Cargar extractor
    print("\n📦 Cargando modelo OSNet...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Usando: {device.upper()}")

    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path=MODEL_PATH,
        device=device
    )

    print("✅ Modelo cargado")

    # Procesar personas
    base_embeddings = []
    statistics = []

    person_folders = [f for f in os.listdir(PERSONAS_BASE)
                     if os.path.isdir(os.path.join(PERSONAS_BASE, f))]

    if not person_folders:
        print(f"\n❌ No se encontraron carpetas de personas en: {PERSONAS_BASE}")
        return

    print(f"\n👥 Personas encontradas: {len(person_folders)}")

    for persona_folder in person_folders:
        persona_dir = os.path.join(PERSONAS_BASE, persona_folder)

        # Parsear nombre y cédula
        partes = persona_folder.split("_")
        if len(partes) >= 3:
            cedula = partes[-1]
            nombre = " ".join(partes[:-1]).replace("_", " ").title()
        else:
            nombre = persona_folder.replace("_", " ").title()
            cedula = "0000000000"

        # Generar embeddings
        result = generate_embeddings_for_person(persona_dir, nombre, extractor)

        if result is not None:
            base_embeddings.append({
                "nombre": nombre,
                "cedula": cedula,
                "embedding": result["embedding"]
            })

            statistics.append({
                "nombre": nombre,
                "n_images": result["n_images"],
                "avg_quality": result["avg_quality"],
                "consistency": result["consistency"],
                "outliers_removed": result.get("outliers_removed", 0)
            })

    # Guardar base de datos
    if base_embeddings:
        with open(BASE_EMBEDDINGS, 'wb') as f:
            pickle.dump(base_embeddings, f)

        print("\n" + "="*70)
        print("📊 RESUMEN GENERAL")
        print("="*70)
        print(f"✅ Personas procesadas: {len(base_embeddings)}")
        print(f"📁 Base guardada en: {BASE_EMBEDDINGS}")

        # Estadísticas generales
        avg_quality_all = np.mean([s["avg_quality"] for s in statistics])
        avg_consistency_all = np.mean([s["consistency"] for s in statistics])
        total_images = sum([s["n_images"] for s in statistics])

        print(f"\n📈 Estadísticas:")
        print(f"   Total de imágenes procesadas: {total_images}")
        print(f"   Calidad promedio general: {avg_quality_all:.1f}%")
        print(f"   Consistencia promedio general: {avg_consistency_all:.1f}%")

        # Detalle por persona
        print(f"\n👥 Detalle por persona:")
        print(f"{'Nombre':<30} {'Imgs':<6} {'Calidad':<10} {'Consistencia':<12}")
        print("-"*70)

        for stat in statistics:
            print(f"{stat['nombre']:<30} "
                  f"{stat['n_images']:<6} "
                  f"{stat['avg_quality']:>6.1f}%   "
                  f"{stat['consistency']:>8.1f}%")

        # Recomendaciones
        print(f"\n💡 Recomendaciones:")

        low_quality = [s for s in statistics if s["avg_quality"] < 60]
        if low_quality:
            print(f"\n⚠️  Personas con baja calidad promedio:")
            for s in low_quality:
                print(f"   - {s['nombre']}: {s['avg_quality']:.1f}% - Recapturar recomendado")

        low_consistency = [s for s in statistics if s["consistency"] < 60]
        if low_consistency:
            print(f"\n⚠️  Personas con baja consistencia:")
            for s in low_consistency:
                print(f"   - {s['nombre']}: {s['consistency']:.1f}% - Verificar variedad de poses")

        few_images = [s for s in statistics if s["n_images"] < 8]
        if few_images:
            print(f"\n⚠️  Personas con pocas imágenes:")
            for s in few_images:
                print(f"   - {s['nombre']}: {s['n_images']} imágenes - Capturar más recomendado")

        if not (low_quality or low_consistency or few_images):
            print(f"   ✨ ¡Excelente! Todas las personas tienen buena calidad")

        # Evaluar modelo con matriz de confusión
        evaluar_modelo(base_embeddings, PERSONAS_BASE, extractor, threshold=0.70)

        print("\n" + "="*70)
        print("✅ PROCESO COMPLETADO")
        print("="*70)
        print("\n💡 Siguiente paso:")
        print("   python deteccion_reid_dron_contador.py")

    else:
        print("\n❌ No se pudo procesar ninguna persona")

if __name__ == "__main__":
    main()
