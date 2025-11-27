# 🚁 Sistema Completo de Re-Identificación con Conteo para Dron DJI FPV

## 📌 Resumen del Sistema

Sistema avanzado de detección, re-identificación y conteo de personas optimizado para video de drones DJI FPV. Incluye tracking robusto, conteo de personas únicas, y exportación de estadísticas detalladas.

---

## 🎯 Características Principales

### ✅ Re-Identificación Mejorada
- Tracking persistente con IDs únicos
- Cuadros de detección permanecen 15 frames (sin parpadeo)
- Sistema de votación para identificaciones estables
- Suavizado de bounding boxes (compensa vibraciones)
- Optimizado para movimiento de cámara de dron

### ✅ Sistema de Conteo
- **Conteo actual**: Personas en área visible ahora
- **Conteo único**: Personas diferentes detectadas (no duplica)
- **Estadísticas avanzadas**: Pico, promedios, historial
- **Sistema de zonas**: Conteo por sectores (opcional)
- **Exportación**: JSON y CSV con datos completos

### ✅ Visualización
- Panel de estadísticas en tiempo real
- Barra de confianza visual
- Colores distintivos (verde=identificado, naranja=desconocido)
- FPS y métricas en pantalla
- Zonas semitransparentes (opcional)

---

## 📂 Estructura de Archivos

```
C:\tesis\
├── app/
│   ├── deteccion_reid_dron.py              # ReID optimizado (sin conteo)
│   ├── deteccion_reid_dron_contador.py     # ⭐ ReID + Conteo completo
│   ├── test_dron_webcam.py                 # Prueba rápida con webcam
│   ├── config_dron.py                      # Configuración centralizada
│   ├── analizar_estadisticas.py            # Análisis de datos guardados
│   ├── [scripts originales...]
│   └── data/
│       ├── personas_base/                  # Fotos de referencia
│       ├── base_personas.pkl               # Embeddings
│       └── estadisticas/                   # ⭐ Estadísticas guardadas
│           ├── stats_YYYYMMDD_HHMMSS.json
│           ├── stats_YYYYMMDD_HHMMSS.csv
│           └── ...
├── GUIA_DRON.md                            # Guía de re-identificación
├── GUIA_CONTEO.md                          # ⭐ Guía del sistema de conteo
└── README_SISTEMA_COMPLETO.md              # Este archivo
```

---

## 🚀 Inicio Rápido

### Opción 1: Sistema Completo (ReID + Conteo)

```bash
cd C:\tesis\app
python deteccion_reid_dron_contador.py
```

**Qué hace:**
- Detecta y re-identifica personas
- Cuenta total en área y personas únicas
- Exporta estadísticas automáticamente
- Muestra panel de conteo en pantalla

### Opción 2: Solo Re-Identificación (Sin Conteo)

```bash
python deteccion_reid_dron.py
```

### Opción 3: Prueba con Webcam

```bash
python test_dron_webcam.py
```

---

## 🎛️ Configuración Rápida

### Archivo: `config_dron.py`

#### 1. Seleccionar Preset

```python
# Descomentar según tu caso de uso:
Presets.apply_balanced()       # ✅ Recomendado para DJI FPV
# Presets.apply_high_quality()   # Vuelo lento, mejor calidad
# Presets.apply_high_speed()     # Vuelo rápido, más FPS
# Presets.apply_fpv_racing()     # Racing extremo
```

#### 2. Configurar Fuente de Video

```python
class VideoConfig:
    VIDEO_SOURCE = "rtmp://localhost:1935/live/stream"  # RTMP
    # VIDEO_SOURCE = 0                                  # Webcam
    # VIDEO_SOURCE = "C:/videos/vuelo.mp4"              # Video
```

#### 3. Activar/Desactivar Conteo

```python
class CountingConfig:
    ENABLE_COUNTING = True  # True = con conteo, False = sin conteo
    SAVE_STATS_INTERVAL = 60  # Guardar cada 60 seg (0 = desactivar)
    EXPORT_FORMAT = "both"  # "csv", "json", "both"
```

#### 4. Configurar Zonas (Opcional)

```python
class ZonesConfig:
    ENABLE_ZONES = True  # Activar zonas

    ZONES = {
        "entrada": {"x1": 0, "y1": 0, "x2": 640, "y2": 720},
        "salida": {"x1": 640, "y1": 0, "x2": 1280, "y2": 720}
    }
```

---

## 📊 Panel de Conteo en Pantalla

```
┌─────────────────────────────────────┐
│ CONTEO DE PERSONAS                 │
├─────────────────────────────────────┤
│ EN AREA ACTUAL:                    │
│   Total:                 5         │
│   Identificados:         3         │
│   Desconocidos:          2         │
│                                    │
│ PERSONAS UNICAS:                   │
│   Total vistas:          25        │
│   Identificadas:         15        │
│   Desconocidas:          10        │
│                                    │
│ ESTADISTICAS:                      │
│   FPS:                   28.5      │
│   Pico personas:         12        │
│                                    │
│ CONTEO POR ZONAS:      (opcional)  │
│   entrada:              2          │
│   salida:               3          │
└─────────────────────────────────────┘
```

---

## ⌨️ Controles

### Durante Ejecución

| Tecla | Acción |
|-------|--------|
| **q** | Salir y guardar estadísticas finales |
| **s** | Guardar estadísticas manualmente ahora |
| **r** | Resetear todos los contadores |
| **ESC** | Salir (alternativo a 'q') |

---

## 📈 Análisis de Estadísticas

### Ver Estadísticas Guardadas

```bash
python analizar_estadisticas.py
```

**Funcionalidades:**
1. Lista todos los archivos de estadísticas
2. Muestra resumen detallado
3. Genera gráficas (requiere matplotlib)
4. Exporta reporte en texto

### Ejemplo de Salida

```
📊 RESUMEN DE SESIÓN
==================================================================
⏱️  Duración: 450.5 segundos
👥 Total personas únicas detectadas: 25
✅ Personas identificadas: 15
   - Juan Perez
   - Maria Lopez
   - Carlos Rodriguez
   ...
❓ Personas desconocidas: 10
📈 Pico de personas simultáneas: 12
```

---

## 📊 Tipos de Métricas

### 1. Conteo Actual (En Área)
**Responde:** ¿Cuántas personas hay AHORA?

```
Total en área: 5
```

### 2. Conteo Único
**Responde:** ¿Cuántas personas DIFERENTES han pasado?

```
Personas únicas vistas: 25
```

- No duplica
- Cada persona cuenta solo una vez
- Usa tracking IDs

### 3. Pico
**Responde:** ¿Cuál fue el máximo simultáneo?

```
Pico: 12 personas
```

### 4. Por Zona
**Responde:** ¿Cuántas personas hay en cada sector?

```
Zona entrada: 2
Zona salida: 3
```

---

## 📁 Archivos de Estadísticas

### Formato JSON (Completo)

**Archivo:** `stats_20241127_153045.json`

**Contiene:**
- Resumen de sesión
- Historial temporal (cada ~1 seg)
- Log detallado de cada detección
- Lista de nombres identificados
- Conteo por zonas (si activo)

### Formato CSV (Resumen)

**Archivo:** `stats_20241127_153045.csv`

**Contiene:**
- Tabla de métricas principales
- Fácil de importar en Excel

---

## 🎯 Casos de Uso

### Caso 1: Vigilancia de Perímetro
```python
# config_dron.py
Presets.apply_balanced()
CountingConfig.ENABLE_COUNTING = True
ZonesConfig.ENABLE_ZONES = False
```

**Métricas clave:**
- Personas únicas vistas
- Personas desconocidas

### Caso 2: Control de Aforo en Evento
```python
Presets.apply_high_quality()
CountingConfig.ENABLE_COUNTING = True
```

**Métricas clave:**
- Total en área actual
- Pico de personas

### Caso 3: Análisis de Flujo (Entrada/Salida)
```python
Presets.apply_balanced()
ZonesConfig.ENABLE_ZONES = True
ZonesConfig.ZONES = {
    "entrada": {...},
    "salida": {...}
}
```

**Métricas clave:**
- Conteo por zona
- Historial temporal

---

## 🔧 Ajustes Comunes

### Cuadros Desaparecen Muy Rápido
```python
class TrackingConfig:
    TRACK_PERSISTENCE_FRAMES = 25  # Aumentar de 15
    MAX_TRACK_AGE = 45
```

### Tracking se Pierde con Movimiento
```python
class TrackingConfig:
    IOU_THRESHOLD = 0.2  # Reducir de 0.3
```

### Identificaciones Inestables (Parpadeo)
```python
class SmoothingConfig:
    VOTE_WINDOW = 5  # Aumentar de 3

class ReIDConfig:
    REID_UPDATE_INTERVAL = 7  # Aumentar de 5
```

### Mejorar FPS
```python
class VideoConfig:
    SCALE_PERCENT = 40  # Reducir de 60
    FRAME_SKIP = 2

class DetectionConfig:
    IMGSZ = 320  # Reducir de 640
```

---

## 📚 Documentación Completa

### Guías Disponibles

| Archivo | Contenido |
|---------|-----------|
| `GUIA_DRON.md` | Guía completa de re-identificación para drones |
| `GUIA_CONTEO.md` | Guía detallada del sistema de conteo |
| `document.md` | Documentación original del proyecto |

### Temas Cubiertos

**GUIA_DRON.md:**
- Mejoras implementadas para drones
- Configuración de presets
- Ajuste de parámetros
- Solución de problemas
- Consejos para vuelo

**GUIA_CONTEO.md:**
- Tipos de conteo
- Sistema de zonas
- Exportación de datos
- Análisis de estadísticas
- Casos de uso detallados

---

## 🆚 Comparación de Scripts

| Script | ReID | Tracking | Conteo | Zonas | Export |
|--------|------|----------|--------|-------|--------|
| `deteccion_reid.py` (original) | ✅ | ⚠️ Básico | ❌ | ❌ | ❌ |
| `deteccion_reid_dron.py` | ✅ | ✅ Robusto | ❌ | ❌ | ❌ |
| `deteccion_reid_dron_contador.py` | ✅ | ✅ Robusto | ✅ | ✅ | ✅ |
| `test_dron_webcam.py` | ✅ | ✅ Robusto | ❌ | ❌ | ❌ |

**Recomendación:** Usar `deteccion_reid_dron_contador.py` para funcionalidad completa.

---

## ⚡ Resumen de Mejoras Implementadas

### Mejoras de Re-Identificación
✅ Sistema de tracking con IoU
✅ Persistencia temporal de 15 frames
✅ Votación de identificaciones (ventana de 3 frames)
✅ Suavizado de bounding boxes (promedio de 5 frames)
✅ IDs persistentes únicos
✅ Barra de confianza visual
✅ Optimización para movimiento de dron

### Mejoras de Conteo
✅ Conteo en tiempo real
✅ Conteo de personas únicas (sin duplicar)
✅ Sistema de zonas configurable
✅ Exportación automática (JSON + CSV)
✅ Historial temporal detallado
✅ Panel de estadísticas en pantalla
✅ Script de análisis de datos
✅ Generación de gráficas

### Mejoras de Configuración
✅ 4 presets predefinidos
✅ Configuración modular y clara
✅ Documentación completa
✅ Scripts de prueba

---

## 🎓 Flujo de Trabajo Completo

### 1. Preparación
```bash
# Verificar que base de datos existe
dir C:\tesis\app\data\base_personas.pkl

# Si no existe, generar:
python generar_base_personas.py
```

### 2. Configuración
```python
# Editar config_dron.py
Presets.apply_balanced()  # Seleccionar preset
VideoConfig.VIDEO_SOURCE = "..."  # Configurar fuente
CountingConfig.ENABLE_COUNTING = True  # Activar conteo
```

### 3. Ejecución
```bash
# Iniciar servidor RTMP (si es necesario)
docker-compose up rtmp

# Ejecutar sistema
python deteccion_reid_dron_contador.py
```

### 4. Monitoreo
- Observar panel de conteo en pantalla
- Estadísticas se guardan automáticamente cada 60 seg
- Presionar 's' para guardar manualmente

### 5. Finalización
- Presionar 'q' para salir
- Ver resumen final en terminal
- Estadísticas finales se guardan automáticamente

### 6. Análisis
```bash
# Analizar estadísticas guardadas
python analizar_estadisticas.py

# Seleccionar archivo
# Ver resumen
# Generar gráficas (opcional)
# Exportar reporte (opcional)
```

---

## 🐛 Solución Rápida de Problemas

| Problema | Solución Rápida |
|----------|-----------------|
| Cuadros desaparecen rápido | `TRACK_PERSISTENCE_FRAMES = 25` |
| Tracking se pierde | `IOU_THRESHOLD = 0.2` |
| Parpadeo en identificaciones | `VOTE_WINDOW = 5` |
| FPS bajo | `SCALE_PERCENT = 40`, `IMGSZ = 320` |
| Conteo duplica personas | `MIN_DETECTIONS_TO_CONFIRM = 5` |
| No se conecta a stream | Probar con webcam: `VIDEO_SOURCE = 0` |

---

## 📞 Archivos Clave

### Scripts Principales
- `deteccion_reid_dron_contador.py` - **Sistema completo** ⭐
- `config_dron.py` - Configuración
- `analizar_estadisticas.py` - Análisis de datos

### Documentación
- `GUIA_DRON.md` - Re-identificación
- `GUIA_CONTEO.md` - Sistema de conteo ⭐
- `README_SISTEMA_COMPLETO.md` - Este archivo

### Configuración Original
- `config.py` - Paths de datos
- `requirements.txt` - Dependencias

---

## 🎉 ¡Todo Listo!

Ahora tienes un sistema completo de:
- ✅ Detección de personas
- ✅ Re-identificación robusta
- ✅ Tracking persistente
- ✅ Conteo en tiempo real
- ✅ Conteo de personas únicas
- ✅ Sistema de zonas
- ✅ Exportación de estadísticas
- ✅ Análisis de datos

**Optimizado específicamente para tu dron DJI FPV** 🚁

---

## 📊 Ejemplo de Resultados

```
📊 RESUMEN DE SESIÓN
==================================================================
⏱️  Duración: 1800 segundos (30 minutos)
🎥 Frames procesados: 54000

👥 PERSONAS DETECTADAS:
   Total personas únicas vistas: 127
   Personas identificadas: 35
      - Juan Perez
      - Maria Lopez
      - Carlos Rodriguez
      ... (32 más)
   Personas desconocidas: 92

📈 ACTIVIDAD:
   Pico de personas simultáneas: 23
   Momento del pico: 2024-11-27T15:45:23
   Promedio en área: 8.5 personas

🗺️  POR ZONA:
   Zona Norte: 45 personas
   Zona Sur: 38 personas
   Zona Este: 24 personas
   Zona Oeste: 20 personas
==================================================================
📁 Estadísticas guardadas en:
   C:\tesis\app\data\estadisticas\stats_20241127_154530.json
   C:\tesis\app\data\estadisticas\stats_20241127_154530.csv
```

---

**¡Listo para volar! 🚁✨**
