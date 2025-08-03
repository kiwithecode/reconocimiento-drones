# 📚 Documentación del Sistema de Re-Identificación en Tiempo Real

Este proyecto implementa un sistema de re-identificación de personas en tiempo real utilizando YOLOv8 para detección y OSNet para extracción de embeddings, con una base local de personas. El sistema es capaz de identificar personas desde una webcam o un stream RTMP.

---

## 📦 Estructura de carpetas

```
tesis/
├── personas_base/                 # Carpeta con subcarpetas por persona (nombre_cedula)
│   └── kevin_armas_1726414087/
│       ├── 1726414087_Kevin_Armas_1.jpg
│       ├── ...
├── base_personas.pkl             # Base generada de embeddings
├── osnet_x1_0_imagenet.pth       # Pesos del modelo OSNet (se descarga automáticamente)
├── yolov8n.pt                    # Modelo YOLOv8 nano (se descarga automáticamente)
├── generar_base_personas.py      # Genera base de datos de embeddings
├── deteccion_reid.py             # Hace inferencia en tiempo real (cam o RTMP)
├── generar_embeddings.py         # Captura embeddings desde webcam
├── renombrar_fotos.py            # Renombra imágenes capturadas para una persona
├── capturar_persona.py           # Captura fotos desde la webcam
└── deep-person-reid/             # Librería clonada torchreid
```

---

## 🚀 Requisitos previos

- Python 3.10+ (Windows)
- CUDA (si usas GPU)
- Git

---

## 🔧 Instalación de `deep-person-reid`

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
python -m venv reid-env
reid-env\Scripts\activate
pip install -r requirements.txt
pip install cython
python setup.py install
```

✅ Esto instalará el paquete `torchreid` y su extractor de características.

> Recomendación: puedes instalar `deep-person-reid` dentro de la carpeta `tesis/` para mantener todo junto.

---

## 📸 Captura de imágenes

```bash
python capturar_persona.py
```

- Esto guardará 10 fotos en `personas_base/<nombre>_<cedula>/`.

Opcionalmente, renombra automáticamente con:

```bash
python renombrar_fotos.py
```

---

## 🧠 Generar la base de embeddings

```bash
python generar_base_personas.py
```

- Lee imágenes de `personas_base/`, extrae embeddings y guarda `base_personas.pkl`

---

## 📡 Alzar servidor de transmisión RTMP (local)

### 1. Clonar e instalar NGINX con módulo RTMP

```bash
git clone https://github.com/sergey-dryabzhinsky/nginx-rtmp-win32.git
cd nginx-rtmp-win32
```

### 2. Configurar `nginx.conf`

```nginx
rtmp {
    server {
        listen 1935;
        chunk_size 4096;

        application live {
            live on;
            record off;
        }
    }
}

http {
    server {
        listen 8080;
        location /stat {
            rtmp_stat all;
            rtmp_stat_stylesheet stat.xsl;
        }
        location /stat.xsl {
            root html;
        }
    }
}
```

### 3. Ejecutar servidor NGINX

```bash
start nginx.exe
```

### 4. Transmitir desde OBS u otro software

```
Servidor: rtmp://localhost:1935/live
Stream key: stream
```

### 5. Verificar estado del stream:

```
http://localhost:8080/stat
```

---

## 🎯 Detección y re-identificación (RTMP o webcam)

```bash
python deteccion_reid.py
```

El script muestra en tiempo real la detección e identificación de personas. Puedes cambiar la fuente de video (RTMP o webcam) directamente en el script.

---

## 📦 Scripts y su propósito

| Script                     | Propósito                                                             |
| -------------------------- | --------------------------------------------------------------------- |
| `capturar_persona.py`      | Captura fotos desde webcam y guarda en carpeta local                  |
| `renombrar_fotos.py`       | Renombra automáticamente las imágenes capturadas                      |
| `generar_base_personas.py` | Genera la base `.pkl` con los embeddings para identificar             |
| `deteccion_reid.py`        | Detecta e identifica personas en video en tiempo real                 |
| `generar_embeddings.py`    | Extrae embeddings desde webcam y guarda el `.pkl` base de una persona |

---

## ✅ Ejecución recomendada paso a paso

```bash
# 1. Capturar fotos desde webcam
python capturar_persona.py

# 2. Renombrar automáticamente (opcional)
python renombrar_fotos.py

# 3. Generar base de embeddings
python generar_base_personas.py

# 4. Levantar RTMP con nginx (si usas transmisión)
start nginx.exe

# 5. Ejecutar detección y re-identificación
python deteccion_reid.py
```

---

# Construir y subir imagen (ej. Docker Hub)

docker build -t tuusuario/reid-app:latest .
docker push tuusuario/reid-app:latest

# Desplegar en K8s

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
