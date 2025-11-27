# 📸 Guía de Captura y Embeddings Mejorados

## 🎯 Mejoras Implementadas

### Scripts Mejorados Creados

1. **`capturar_persona_mejorado.py`** - Captura inteligente con validación de calidad
2. **`generar_base_personas_mejorado.py`** - Generación de embeddings con data augmentation

---

## 🚀 Comparación: Original vs Mejorado

### Script de Captura

| Característica | Original | Mejorado |
|----------------|----------|----------|
| Validación de calidad | ❌ | ✅ Nitidez, brillo, tamaño |
| Feedback visual | Básico | ✅ Indicadores de calidad en tiempo real |
| Evita duplicados | ❌ | ✅ Compara similitud entre capturas |
| Captura inteligente | Manual/Timer simple | ✅ Selecciona mejores frames |
| Diversidad | No verifica | ✅ Fuerza variedad de poses |
| Estadísticas | Básicas | ✅ Detalladas con calidad promedio |

### Script de Embeddings

| Característica | Original | Mejorado |
|----------------|----------|----------|
| Data augmentation | ❌ | ✅ Flip, brillo, contraste |
| Filtrado de outliers | ❌ | ✅ Distancia de Mahalanobis |
| Promedio | Simple | ✅ Ponderado por calidad |
| Normalización | ❌ | ✅ L2 normalization |
| Validación de consistencia | ❌ | ✅ Score de consistencia |
| Reporte | Básico | ✅ Detallado con recomendaciones |

---

## 📋 Flujo de Trabajo Mejorado

### Paso 1: Captura con Validación de Calidad

```bash
cd C:\tesis\app
python capturar_persona_mejorado.py
```

#### Durante la Captura

**Interfaz Visual:**
```
┌─────────────────────────────────────┐
│ Capturas: 12/15                    │
│ Calidad promedio: 85.3%            │
│ Progreso: 80%                      │
│ Modo: AUTO                         │
│                                    │
│ [████████████████░░░░] 80%         │
│                                    │
│ Calidad actual:                    │
│ EXCELENTE (92%)                    │
│ [████████████████████] 92%         │
└─────────────────────────────────────┘
```

**Indicadores de Calidad:**
- 🟢 **Verde (80-100%)**: EXCELENTE - Calidad óptima
- 🟡 **Amarillo (60-79%)**: BUENA - Aceptable
- 🟠 **Naranja (40-59%)**: REGULAR - Mejorar condiciones
- 🔴 **Rojo (0-39%)**: MALA - No guardar

**Controles:**
- **'s'**: Guardar manualmente (solo si calidad ≥ 50%)
- **'a'**: Activar/desactivar modo automático
- **'q'**: Salir

#### Modo Automático Inteligente

Cuando activas modo AUTO (tecla 'a'), el sistema:

1. **Valida nitidez**: Descarta imágenes borrosas
2. **Verifica brillo**: Rechaza muy oscuras o sobreexpuestas
3. **Compara similitud**: Evita capturas casi idénticas
4. **Espaciado temporal**: Mínimo 1.5 seg entre capturas
5. **Selecciona mejores frames**: Solo guarda calidad ≥ 70%

#### Consejos para Buena Captura

✅ **Hacer:**
- Variar el ángulo de la cámara/dron
- Diferentes poses (de frente, 3/4, perfil)
- Buena iluminación natural
- Persona de pie, cuerpo completo visible
- Mantener distancia 3-8 metros

❌ **Evitar:**
- Luz directa por detrás (contraluz)
- Movimiento rápido (blur)
- Objetos tapando a la persona
- Sombras muy fuertes
- Imágenes muy oscuras

#### Resumen al Finalizar

```
📊 RESUMEN DE CAPTURA
==================================================================
✅ Total de imágenes capturadas: 15
📈 Calidad promedio: 87.2%
📁 Guardadas en: C:\tesis\app\data\personas_base\Juan_Perez_0123456789

✨ EXCELENTE: 15 imágenes capturadas

💡 Siguiente paso:
   python generar_base_personas_mejorado.py
==================================================================
```

---

### Paso 2: Generar Embeddings con Data Augmentation

```bash
python generar_base_personas_mejorado.py
```

#### Configuración por Defecto

```python
# En el script, puedes modificar:
USE_AUGMENTATION = True        # Data augmentation
FILTER_OUTLIERS = True         # Filtrar outliers
USE_WEIGHTED_AVERAGE = True    # Promedio ponderado
NORMALIZE_EMBEDDINGS = True    # Normalización L2
```

#### Proceso por Persona

```
======================================================================
Procesando: Juan Perez
======================================================================
📸 Imágenes encontradas: 15
🧠 Embeddings generados: 60
   (incluyendo augmentation)
🔍 Outliers removidos: 3/60
📊 Consistencia: 92.5%
⚖️  Promedio ponderado por calidad
🔧 Embedding normalizado (L2)
✅ Embedding final generado
   Calidad promedio: 87.2%
   Consistencia: 92.5%
```

**Qué significa cada métrica:**

- **Embeddings generados**: Original + augmentados (4x si augmentation activo)
- **Outliers removidos**: Embeddings anómalos descartados
- **Consistencia**: Qué tan similares son los embeddings entre sí (100% = idénticos)
- **Calidad promedio**: Promedio de calidad de las imágenes fuente

#### Data Augmentation

Para cada imagen original, se generan versiones adicionales:

1. **Original**: Sin modificar
2. **Flip horizontal**: Espejo
3. **Más brillante**: +20% brillo
4. **Más oscuro**: -20% brillo

Esto genera **4x más embeddings**, mejorando robustez.

#### Filtrado de Outliers

Usa **distancia de Mahalanobis** para detectar embeddings anómalos:
- Calcula centroide de todos los embeddings
- Mide distancia estadística de cada uno
- Descarta los que están > 2.5 desviaciones estándar

**Beneficio**: Elimina errores de captura o detección.

#### Promedio Ponderado

En lugar de promedio simple, pondera por calidad:

```
Peso = (calidad / 100) ^ 2

Imagen con 90% calidad → peso 0.81
Imagen con 60% calidad → peso 0.36
```

**Beneficio**: Imágenes de mejor calidad tienen más influencia.

#### Normalización L2

Normaliza el vector embedding final:

```
embedding_normalizado = embedding / ||embedding||
```

**Beneficio**: Mejora comparaciones por similitud coseno.

---

### Resumen Final

```
======================================================================
📊 RESUMEN GENERAL
======================================================================
✅ Personas procesadas: 3
📁 Base guardada en: C:\tesis\app\data\base_personas.pkl

📈 Estadísticas:
   Total de imágenes procesadas: 45
   Calidad promedio general: 85.7%
   Consistencia promedio general: 88.3%

👥 Detalle por persona:
Nombre                         Imgs   Calidad    Consistencia
----------------------------------------------------------------------
Juan Perez                     15     87.2%       92.5%
Maria Lopez                    18     90.5%       95.1%
Carlos Rodriguez               12     79.3%       77.8%

💡 Recomendaciones:

⚠️  Personas con baja consistencia:
   - Carlos Rodriguez: 77.8% - Verificar variedad de poses

✅ PROCESO COMPLETADO
======================================================================

💡 Siguiente paso:
   python deteccion_reid_dron_contador.py
```

---

## 🔧 Ajustes de Configuración

### Captura más Estricta (Máxima Calidad)

Edita `capturar_persona_mejorado.py`:

```python
MIN_SHARPNESS = 150           # Aumentar de 100
MIN_DETECTION_CONFIDENCE = 0.7  # Aumentar de 0.6
SIMILARITY_THRESHOLD = 0.90    # Aumentar de 0.85
```

### Captura más Permisiva (Condiciones Difíciles)

```python
MIN_SHARPNESS = 70            # Reducir de 100
MIN_DETECTION_CONFIDENCE = 0.4  # Reducir de 0.6
SIMILARITY_THRESHOLD = 0.70    # Reducir de 0.85
```

### Más Data Augmentation

Edita `generar_base_personas_mejorado.py`:

```python
# En la función augment_image(), agregar:

# Rotación leve
rotated = cv2.warpAffine(image, M, (w, h))
augmented.append(rotated)

# Cambio de contraste
contrast = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
augmented.append(contrast)
```

### Desactivar Filtrado de Outliers

```python
FILTER_OUTLIERS = False
```

**Cuándo desactivar:**
- Pocas imágenes (< 5)
- Todas las imágenes son de buena calidad
- Quieres incluir todo

---

## 📊 Interpretación de Métricas

### Calidad de Imagen (0-100%)

| Rango | Significado | Acción |
|-------|-------------|--------|
| 90-100% | Excelente | ✅ Perfecta |
| 70-89% | Buena | ✅ Aceptable |
| 50-69% | Regular | ⚠️ Mejorar condiciones |
| 0-49% | Mala | ❌ No usar |

**Factores que afectan calidad:**
- Nitidez (más importante)
- Brillo/iluminación
- Tamaño de la persona en frame

### Consistencia de Embeddings (0-100%)

| Rango | Significado | Causa Probable |
|-------|-------------|----------------|
| 90-100% | Excelente | Capturas muy similares |
| 75-89% | Buena | Variedad moderada de poses |
| 60-74% | Regular | Mucha variedad o cambios de ropa |
| 0-59% | Baja | Diferentes personas, errores, o extrema variedad |

**Alta consistencia (>90%):**
- ✅ Bueno: Todas las capturas son de la misma persona en condiciones similares
- ⚠️ Problema potencial: Falta variedad (todas de frente, mismo ángulo)

**Baja consistencia (<60%):**
- ❌ Problema: Verificar que todas las imágenes son de la misma persona
- ❌ Problema: Cambios drásticos (ropa diferente, ángulos extremos)

### Número de Imágenes

| Cantidad | Evaluación | Precisión Esperada |
|----------|------------|-------------------|
| 15+ | Óptimo | 85-95% |
| 10-14 | Bueno | 75-85% |
| 8-9 | Aceptable | 65-75% |
| 5-7 | Mínimo | 55-65% |
| <5 | Insuficiente | <55% |

---

## 🎯 Casos de Uso Específicos

### Caso 1: Captura desde Dron en Movimiento

**Desafío**: Dron en movimiento, vibraciones, ángulos variables

**Configuración recomendada:**

```python
# capturar_persona_mejorado.py
MIN_SHARPNESS = 80  # Más permisivo (hay vibración)
SIMILARITY_THRESHOLD = 0.80  # Más diversidad (movimiento)
MIN_DISTANCE_BETWEEN_CAPTURES = 2.0  # Más espacio temporal
```

**Proceso:**
1. Volar en círculo alrededor de la persona (3-5 metros)
2. Modo AUTO activado
3. Capturar desde diferentes ángulos (0°, 45°, 90°, 135°, 180°)
4. Mínimo 15 imágenes

### Caso 2: Captura Estacionaria (Trípode/Dron Hover)

**Desafío**: Poca variedad de ángulos

**Configuración recomendada:**

```python
MIN_SHARPNESS = 120  # Más estricto (cámara estable)
SIMILARITY_THRESHOLD = 0.90  # Evitar duplicados
```

**Proceso:**
1. Persona se mueve y cambia poses
2. Diferentes expresiones
3. Modo MANUAL para seleccionar mejores momentos

### Caso 3: Condiciones de Baja Luz

**Desafío**: Poca iluminación, ISO alto, ruido

**Configuración recomendada:**

```python
MIN_SHARPNESS = 60  # Muy permisivo
is_image_too_dark(image, threshold=30)  # Más permisivo
```

**Proceso:**
1. Usar iluminación adicional si es posible
2. Capturar más imágenes (20+) para compensar
3. Modo MANUAL para seleccionar mejores frames

---

## 🐛 Solución de Problemas

### ❌ "Muy similar a captura reciente"

**Causa**: Modo AUTO detecta que la imagen es casi idéntica a una anterior

**Solución:**
- Pedir a la persona que cambie de pose
- Cambiar ángulo de la cámara/dron
- Esperar más tiempo entre capturas
- Reducir `SIMILARITY_THRESHOLD` en configuración

### ❌ "Calidad baja: Borrosa"

**Causa**: Imagen sin nitidez

**Solución:**
- Estabilizar cámara/dron
- Activar gimbal
- Mejor iluminación
- Reducir velocidad de movimiento
- Enfocar manualmente

### ❌ "Calidad baja: Muy oscura"

**Causa**: Poca iluminación

**Solución:**
- Agregar luz
- Ajustar configuración de cámara (ISO, apertura)
- Cambiar ubicación (más luz natural)
- Reducir umbral en configuración

### ❌ "No se puede guardar: Muy pequeña"

**Causa**: Persona muy lejos o muy cerca del borde

**Solución:**
- Acercar el dron (3-8 metros óptimo)
- Centrar a la persona en el frame
- Verificar que cuerpo completo esté visible

### ❌ Consistencia muy baja (<60%)

**Causa**: Embeddings muy diferentes entre sí

**Solución:**
1. Verificar que todas las imágenes son de la misma persona
2. Revisar si hay errores de detección (objetos confundidos con personas)
3. Eliminar imágenes con ropa completamente diferente
4. Recapturar con condiciones más uniformes

### ❌ Muchos outliers removidos (>30%)

**Causa**: Muchas imágenes anómalas

**Solución:**
1. Revisar calidad de captura
2. Verificar que YOLO detectó correctamente
3. Eliminar imágenes problemáticas manualmente
4. Recapturar si es necesario

---

## 📈 Mejores Prácticas

### ✅ Captura Óptima

1. **Iluminación**: Luz día difusa, evitar sombras fuertes
2. **Distancia**: 3-8 metros del sujeto
3. **Ángulos**: Variar entre 0° (frontal) y 90° (lateral)
4. **Poses**: De frente, 3/4 izquierda, 3/4 derecha, perfil
5. **Cantidad**: 15-20 imágenes por persona
6. **Velocidad**: Movimiento lento y suave del dron
7. **Estabilidad**: Gimbal activado, modo suave

### ✅ Generación de Embeddings

1. **Revisar capturas**: Eliminar imágenes claramente malas antes
2. **Usar augmentation**: Siempre activado (mejora robustez)
3. **Filtrar outliers**: Activado si tienes ≥8 imágenes
4. **Normalización**: Siempre activada
5. **Promedio ponderado**: Activado si calidades varían mucho

---

## 🔬 Validación del Sistema

### Después de Generar Embeddings

**Verifica:**

1. **Consistencia ≥ 75%**: Buena homogeneidad
2. **Calidad promedio ≥ 70%**: Buenas imágenes fuente
3. **Mínimo 8 imágenes**: Por persona
4. **Outliers < 20%**: Mayoría de embeddings válidos

### Prueba de Funcionamiento

```bash
# Ejecutar sistema completo
python deteccion_reid_dron_contador.py

# Verificar:
1. Personas conocidas se identifican correctamente (> 70% confianza)
2. Poco parpadeo en identificación (gracias a votación)
3. Funciona desde diferentes ángulos
4. Funciona con cambios de luz
```

---

## 📚 Comparación de Scripts

| Tarea | Script Original | Script Mejorado | Cuándo Usar Mejorado |
|-------|----------------|-----------------|----------------------|
| Captura | `capturar_persona.py` | `capturar_persona_mejorado.py` | ✅ Siempre (mejor calidad) |
| Embeddings | `generar_base_personas.py` | `generar_base_personas_mejorado.py` | ✅ Siempre (mejor precisión) |

**Recomendación**: Usar siempre las versiones mejoradas para producción.

---

## 🎓 Resumen

### Mejoras Clave

1. **Validación de calidad en tiempo real** - Solo guarda imágenes buenas
2. **Captura inteligente** - Evita duplicados y selecciona mejores frames
3. **Data augmentation** - 4x más embeddings para robustez
4. **Filtrado de outliers** - Elimina errores automáticamente
5. **Promedio ponderado** - Mejor peso a imágenes de calidad
6. **Normalización** - Mejora comparaciones

### Flujo Completo

```
1. python capturar_persona_mejorado.py
   ↓ 15+ imágenes con calidad validada

2. python generar_base_personas_mejorado.py
   ↓ Embeddings con augmentation y filtrado

3. python deteccion_reid_dron_contador.py
   ↓ Sistema completo con re-ID precisa
```

---

¡Listo! Ahora tienes un sistema de captura y generación de embeddings de alta calidad. 📸✨
