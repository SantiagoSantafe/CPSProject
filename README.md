Aquí tienes el código fuente en formato Markdown listo para copiar y pegar en tu archivo `README.md`. He mantenido un tono profesional, técnico y limpio, eliminando emojis innecesarios y optimizando la estructura.

```markdown
# Language-Grounded 3D Semantic Mapping and Navigation

Este proyecto implementa un **pipeline modular para percepción de vocabulario abierto, mapeo semántico 3D y navegación guiada por lenguaje natural**. El sistema permite a un agente robótico detectar objetos mediante descripciones textuales, integrarlos en un mapa semántico tridimensional y generar trayectorias de navegación basadas en comandos verbales.

Un objetivo central del diseño es la **reproducibilidad**: el pipeline completo puede ejecutarse sin necesidad de GPU, ROS o sensores físicos mediante un **modo dry-run**, mientras que el mismo núcleo de código es compatible con cámaras RGB-D y sensores LiDAR para despliegues reales.

---

## Características Principales

* **Percepción Open-Vocabulary:** Detección de objetos mediante SAM + CLIP (GroundingDINO opcional).
* **Mapeo Semántico:** Proyección de datos RGB-D a 3D y fusión de objetos con tracking persistente.
* **Procesamiento de Lenguaje:** Motor de consultas semánticas y parsing de comandos naturales.
* **Navegación:** Generación de objetivos espaciales $(x, y, \theta)$ basados en el contexto del mapa.
* **Evaluación Cuantitativa:** Herramientas integradas para medir precisión de recuperación (retrieval) y métricas de navegación.
* **Arquitectura Agnóstica:** Soporte para simulación, datasets offline y sensores en vivo.

---

## Estructura del Repositorio

```text
CPSProject/
├── scripts/                  # Puntos de entrada del sistema
│   ├── run_system.py         # Ejecución del pipeline principal
│   ├── evaluate_system.py    # Generación de reportes y métricas
│   └── download_weights.py   # Gestión de modelos pre-entrenados
├── src/                      # Código fuente modular
│   ├── perception/           # Detectores y descriptores visuales
│   ├── mapping/              # Lógica de fusión y gestión 3D
│   └── navigation/           # Motores de búsqueda y control
├── configs/                  # Escenarios de prueba y diccionarios de consultas
├── tests/                    # Pruebas unitarias y de integración
├── requirements.txt          # Dependencias mínimas para CPU
└── documentation.md          # Documentación técnica extendida

```

---

## Instalación

### Requisitos del Sistema

* **SO:** Linux o macOS (recomendado).
* **Python:** 3.10 – 3.13.
* **Hardware:** Mínimo 8GB RAM. GPU opcional para modelos de percepción acelerados.

### Configuración del Entorno

Este proceso instala las dependencias necesarias para el modo de prueba y la lógica central del sistema.

```bash
git clone <repository-url>
cd CPSProject

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

```

### Verificación del Sistema

Ejecute la suite de pruebas para asegurar la integridad de los módulos centrales:

```bash
pytest -m "not gpu and not ros and not slow"

```

---

## Modos de Ejecución

El sistema soporta tres niveles de fidelidad utilizando la misma base de código.

### 1. Modo Dry-Run (Demostración y CI)

Ideal para desarrollo rápido. Utiliza datos sintéticos y stubs de percepción para validar la lógica de mapeo y navegación sin hardware especializado.

```bash
python -m scripts.run_system --dry-run --max-frames 2 --verbose

```

### 2. Datos Offline (Datasets)

Procesa secuencias de imágenes RGB-D pre-grabadas. Requiere la implementación de la interfaz `load_data()` en el script principal para cargar poses e intrínsecos de cámara.

```bash
python -m scripts.run_system --data path/to/dataset --results-dir results/offline_run

```

### 3. Sensores en Vivo (Real-Time)

Para despliegues con hardware como **Orbbec Femto Mega** y **LiDAR**. Requiere dependencias adicionales de Deep Learning y ROS 2.

```bash
# Instalación de dependencias de visión
pip install torch torchvision clip segment-anything

# Ejecución vinculada a tópicos de ROS 2
python -m scripts.run_system

```

---

## Evaluación y Métricas

El módulo de evaluación analiza el rendimiento del sistema basándose en los resultados almacenados en el directorio de salida.

```bash
python scripts/evaluate_system.py --run-dir results/demo_run

```

**Métricas incluidas:**

* **Precisión Top-1 / Top-K:** Efectividad en la localización de objetos consultados.
* **MRR (Mean Reciprocal Rank):** Calidad de la recuperación semántica.
* **Error de Posicionamiento:** Desviación en metros respecto al ground truth.
* **Tasa de Éxito de Navegación:** Porcentaje de objetivos alcanzados satisfactoriamente.

---

## Arquitectura del Sistema

El flujo de información se divide en procesos desacoplados para garantizar la modularidad:

1. **Ingesta:** Captura de flujo RGB-D / LiDAR.
2. **Percepción:** Segmentación y extracción de embeddings semánticos.
3. **Proyección:** Transformación de detecciones 2D a coordenadas mundiales 3D.
4. **Mapeo:** Fusión probabilística de objetos en un mapa global.
5. **Consulta:** Interfaz de lenguaje natural para identificar objetivos.
6. **Control:** Generación de metas de navegación para la base móvil.

---

## Autores

* **Andrés Santiago Santafé Silva**
* **Ana Maria Oliveros Ossa**
