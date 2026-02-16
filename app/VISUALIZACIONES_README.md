# Sistema de Visualizaciones HSV, LBP y HOG

## Descripción

Este sistema genera automáticamente visualizaciones de análisis de características en segundo plano durante la detección y Re-ID de personas. Las visualizaciones se procesan en un thread separado para no afectar el rendimiento del sistema principal.

## Características Generadas

### 1. **HSV (Hue-Saturation-Value)**
- **Propósito**: Análisis de colores de ropa
- **Visualización incluye**:
  - Canal H (Hue/Matiz): Colores dominantes
  - Canal S (Saturation/Saturación): Intensidad del color
  - Canal V (Value/Valor): Brillo
  - Histograma de Hue: Distribución de colores
  - Imagen original para comparación

### 2. **LBP (Local Binary Patterns)**
- **Propósito**: Análisis de texturas y patrones de ropa
- **Visualización incluye**:
  - Imagen original
  - Imagen en escala de grises
  - Mapa LBP con colormap
  - Histograma de patrones LBP
- **Útil para**: Detectar texturas, patrones, rayas, cuadros, etc.

### 3. **HOG (Histogram of Oriented Gradients)**
- **Propósito**: Análisis de forma y contornos
- **Visualización incluye**:
  - Imagen original
  - Imagen en escala de grises
  - Mapa de gradientes direccionales
  - Histograma de características HOG
- **Útil para**: Análisis de silueta, postura, forma corporal

## Estructura de Carpetas

```
app/data/visualizaciones/
├── hsv/          # Visualizaciones HSV
├── lbp/          # Visualizaciones LBP
├── hog/          # Visualizaciones HOG
└── original/     # Imágenes originales (crops)
```

## Nomenclatura de Archivos

Los archivos se guardan con el siguiente formato:

```
{NOMBRE}_track{ID}_frame{FRAME}_{TIMESTAMP}_{TIPO}.png
```

Ejemplo:
```
Juan_Perez_track3_frame450_20260107_143025_123_hsv.png
```

Donde:
- `Juan_Perez`: Identidad de la persona
- `track3`: ID del track
- `frame450`: Número de frame
- `20260107_143025_123`: Timestamp (año-mes-día_hora-min-seg-ms)
- `hsv/lbp/hog`: Tipo de visualización

## Configuración

### Parámetros en `deteccion_reid_dron_contador.py`

```python
# Activar/desactivar sistema completo
ENABLE_VISUALIZATIONS = True

# Frecuencia de guardado (cada 30 frames = ~1 seg a 30fps)
VIZ_SAVE_EVERY_N_FRAMES = 30

# Máximo de visualizaciones por persona
VIZ_MAX_PER_PERSON = 10

# Solo guardar personas identificadas (no desconocidos)
VIZ_ONLY_IDENTIFIED = True

# Confianza mínima para guardar
VIZ_MIN_CONFIDENCE = 0.6
```

### Parámetros Avanzados en `visualization_handler.py`

```python
class VisualizationConfig:
    # Qué visualizaciones generar
    ENABLE_HSV = True
    ENABLE_LBP = True
    ENABLE_HOG = True

    # Tamaño de cola
    QUEUE_MAX_SIZE = 100

    # Formato de imagen
    IMAGE_FORMAT = "png"  # o "jpg"
    IMAGE_QUALITY = 95    # Para JPG

    # Parámetros LBP
    LBP_POINTS = 24
    LBP_RADIUS = 3
    LBP_METHOD = 'uniform'

    # Parámetros HOG
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
```

## Uso

### Ejecución Normal

```bash
cd app
python deteccion_reid_dron_contador.py
```

El sistema de visualizaciones se iniciará automáticamente si `ENABLE_VISUALIZATIONS = True`.

### Controles Durante Ejecución

- **'q'**: Salir y guardar estadísticas
- **'s'**: Guardar estadísticas ahora
- **'r'**: Resetear contadores
- **'v'**: Ver estadísticas de visualizaciones

### Ver Estadísticas en Tiempo Real

Presiona `v` durante la ejecución para ver:
```
======================================================================
🎨 ESTADÍSTICAS DE VISUALIZACIONES
======================================================================
✅ Total procesadas: 25
⏭️  Total omitidas: 5
📦 Cola actual: 2
📊 Guardados por persona:
   - Juan Perez: 10
   - Maria Lopez: 8
   - Carlos Garcia: 7
======================================================================
```

## Rendimiento

### Impacto en FPS

- **Procesamiento en background**: Thread separado, mínimo impacto
- **Cola asíncrona**: No bloquea el bucle principal
- **Filtros inteligentes**: Solo procesa cuando es necesario

### Gestión de Recursos

- **Límite por persona**: Evita saturación de disco
- **Frecuencia controlada**: No procesa todos los frames
- **Cola limitada**: Evita uso excesivo de memoria

## Aplicaciones

### Análisis Posterior

Las visualizaciones generadas son útiles para:

1. **Debugging del sistema Re-ID**
   - Verificar qué características usa el modelo
   - Entender por qué falla la identificación
   - Comparar características entre personas similares

2. **Mejora del modelo**
   - Analizar qué colores/texturas discriminan mejor
   - Identificar casos problemáticos
   - Generar datasets de entrenamiento

3. **Análisis forense**
   - Documentar características de personas detectadas
   - Comparación visual detallada
   - Evidencia de detecciones

4. **Investigación**
   - Estudiar patrones de vestimenta
   - Análisis de multitudes
   - Estadísticas de colores dominantes

## Ejemplo de Salida

Cuando el sistema detecta a "Juan Perez" con confianza 0.85:

1. Se guarda el crop original en `original/`
2. Se genera visualización HSV mostrando colores de su ropa
3. Se genera visualización LBP mostrando texturas (rayas, cuadros, etc.)
4. Se genera visualización HOG mostrando contorno y postura

Cada visualización es una imagen compuesta que combina múltiples análisis en una sola imagen para facilitar la comparación.

## Requisitos

### Dependencias Adicionales

El sistema requiere `scikit-image` para procesamiento LBP y HOG:

```bash
pip install scikit-image==0.24.0
```

O instalar desde requirements.txt:

```bash
pip install -r requirements.txt
```

### Espacio en Disco

Estimar aprox. **2-5 MB** por persona (10 visualizaciones × 3 tipos × ~100KB).

Para 50 personas: ~100-250 MB

## Desactivar el Sistema

Para desactivar completamente las visualizaciones:

```python
# En deteccion_reid_dron_contador.py
ENABLE_VISUALIZATIONS = False
```

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'skimage'"

**Solución**: Instalar scikit-image
```bash
pip install scikit-image
```

### Visualizaciones no se generan

**Verificar**:
1. `ENABLE_VISUALIZATIONS = True`
2. La confianza de detección es >= `VIZ_MIN_CONFIDENCE`
3. Si `VIZ_ONLY_IDENTIFIED = True`, solo se guardan personas identificadas
4. No se ha alcanzado el límite `VIZ_MAX_PER_PERSON`

### Cola se llena (queue_size alto)

**Soluciones**:
- Aumentar `VIZ_SAVE_EVERY_N_FRAMES` (procesar menos frecuentemente)
- Reducir `VIZ_MAX_PER_PERSON` (menos guardados por persona)
- Desactivar algunos tipos: `ENABLE_HOG = False`

### Alto uso de CPU

**Optimizaciones**:
- Desactivar HOG (es el más costoso): `ENABLE_HOG = False`
- Aumentar `VIZ_SAVE_EVERY_N_FRAMES` a 60 o más
- Reducir `LBP_POINTS` a 16

## Licencia

Este módulo es parte del sistema de Re-identificación con contador para dron DJI FPV.
