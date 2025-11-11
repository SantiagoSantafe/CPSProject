# CPS Project Documentation

## Phase 1: Initial Setup & Installation (Oct 23)
This phase involved setting up the ROS 2 Jazzy environment and installing key perception and hardware packages.

### 1.1 ROS 2 & System Packages

**ROS 2 Jazzy Desktop:** Installed using the standard APT repository.

```bash
sudo apt install ros-jazzy-desktop
```

**Camera Driver (Orbbec):** Installed the OrbbecSDK ROS 2 driver for the Femto Mega camera.

Reference: https://github.com/orbbec/OrbbecSDK_ROS2

### 1.2 Perception Libraries (GroundingDINO)

**Problem:** The installation of GroundingDINO failed. The error occurred while pip was trying to "get requirements to build wheel". This is often caused by pip trying to build dependencies in an isolated environment, which can conflict with existing libraries like torch.

**Fix:** We forced pip to use the current environment by using the `--no-build-isolation` flag.

```bash
# 1. Install GroundingDINO without build isolation
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git

# 2. Re-install Segment Anything (which may have failed due to the previous error)
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Phase 2: System Configuration

### 2.1 ROS 2 Environment
To ensure the ROS 2 workspace is always available, the sourcing command was added to the `.bashrc` file for automatic execution.

```bash
# Command added to ~/.bashrc
source ~/ros2_ws/install/setup.bash
```

### 2.2 Launching Camera Node
The Orbbec Femto Mega camera node is launched using its provided launch file.

```bash
ros2 launch orbbec_camera femto_mega.launch.py
```

### 2.3 ROS 2 Bag Utilities
These are the common commands used for recording and replaying experimental data.

**Record all topics:**

```bash
ros2 bag record -a <bag_name>
# Example: ros2 bag record -a lab_environment
```

**Play a bag file (with clock):**

```bash
ros2 bag play <bag_name> --clock
# The --clock flag is critical for simulation time
```

## Phase 3: RTAB-Map Integration & Debugging
This section details the step-by-step troubleshooting process to get RTAB-Map working with our recorded rosbag data.

### Problem 1: Image Resolution Mismatch

**Symptom:** RTAB-Map would not process the bag data. We identified a resolution mismatch between the RGB image (`/camera/color/image_raw`) and the depth image (`/camera/depth/image_raw`).

**Solution:** Create a custom Python ROS 2 node to subscribe to the raw depth image, resize it to match the RGB sensor's resolution (1280x720), and republish it on a new topic.

#### Fix 1.1: resize_depth.py Node
This script was created to perform the resizing.

**File:** `~/resize_depth.py`

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DepthResizer(Node):
    def __init__(self):
        super().__init__('depth_resizer')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',  # Input topic
            self.callback,
            10
        )
        self.pub = self.create_publisher(Image, '/camera/depth/image_resized', 10) # Output topic
        self.get_logger().info('Depth Resizer node started. Subscribing to /camera/depth/image_raw...')

    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # Resize to 1280x720 (matching the RGB sensor)
        resized = cv2.resize(cv_image, (1280, 720), interpolation=cv2.INTER_NEAREST)
        
        out_msg = self.bridge.cv2_to_imgmsg(resized, encoding='passthrough')
        out_msg.header = msg.header
        self.pub.publish(out_msg)

def main():
    rclpy.init()
    node = DepthResizer()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

**Setup:**

```bash
# Make the script executable
chmod +x ~/resize_depth.py

# Install dependencies if missing
sudo apt install ros-jazzy-cv-bridge python3-opencv
```

### Problem 2: Timestamp Synchronization

**Symptom:** After fixing the resolution, RTAB-Map still failed with `[WARN] ... Did not receive data since 5 seconds!` and `[WARN] ... The time difference between rgb and depth frames is high`.

**Cause:** When replaying a rosbag, nodes process data as fast as possible, but the timestamps in the bag are from the past. The system must be told to use the "simulated" clock from the bag file, not the current system time.

**Solution:**
- Play the rosbag using the `--clock` flag.
- Launch RTAB-Map with the `use_sim_time:=true` parameter.

#### Fix 2.1: use_sim_time Workflow
This became the new standard workflow:

**Terminal 1: Play Rosbag**

```bash
ros2 bag play lab_environment --clock
```

**Terminal 2: Run Resize Node**

```bash
python3 ~/resize_depth.py
```

**Terminal 3: Run RTAB-Map**

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
    args:="--delete_db_on_start" \
    depth_topic:=/camera/depth/image_resized \
    rgb_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info \
    depth_camera_info_topic:=/camera/depth/camera_info \
    approx_sync:=true \
    frame_id:=camera_link \
    use_sim_time:=true
```

### Problem 3: Visual Odometry Failure

**Symptom:** Even with correct synchronization, odometry failed. Logs showed `Odom: quality=0`, `[ERROR] ... no odometry is provided`, and `Registration failed: "Not enough inliers 0/20"`.

**Cause:** The default visual odometry strategy (using visual features, Strategy 0) was failing. This was likely due to rapid motion, poor lighting, or lack of texture in the recorded environment.

**Solution:** Switch the odometry strategy from visual features to ICP (Iterative Closest Point) (Strategy 1). ICP uses the 3D point cloud geometry to find the transformation, which is more robust in texture-poor environments.

#### Fix 3.1: Final Working Launch (Using ICP)
This launch command successfully processed the rosbag and generated a map.

**Terminal 3 (Final): Run RTAB-Map with ICP**

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
    args:="--delete_db_on_start --Odom/Strategy 1" \
    depth_topic:=/camera/depth/image_resized \
    rgb_topic:=/camera/color/image_raw \
    camera_info_topic:=/camera/color/camera_info \
    depth_camera_info_topic:=/camera/depth/camera_info \
    approx_sync:=true \
    approx_sync_max_interval:=0.2 \
    frame_id:=camera_link \
    use_sim_time:=true \
    queue_size:=50
```

**Result:** This solution worked.

**Note:** The process was observed to be slow. For future tests, we can also experiment with playing the bag at a slower rate (e.g., `ros2 bag play lab_environment --clock --rate 0.5`) to give the ICP algorithm more time to process each frame.

Análisis Detallado del Mapeo RTAB-Map con Cámara RGB-D Femto Mega
Basándome en las imágenes que has compartido, voy a explicar detalladamente cada elemento del proceso de mapeo SLAM que estás realizando.
🎥 Cámara Femto Mega RGB-D
Esta es una cámara de profundidad que captura simultáneamente:

RGB: Imagen a color del entorno
Depth (D): Información de profundidad de cada píxel, creando una nube de puntos 3D

🗺️ RTAB-Map (Real-Time Appearance-Based Mapping)
Es un algoritmo de SLAM (Simultaneous Localization and Mapping) que permite:

Mapear el entorno en tiempo real
Localizar la posición del robot/cámara simultáneamente
Detectar cierres de bucle (loop closures) para corregir la deriva del mapa


📊 Análisis de los Elementos Visuales
1. Nube de Puntos (Point Cloud) - Panel Superior Derecho
Las nubes de puntos que ves representan el espacio 3D capturado:

Puntos blancos/grises: Superficie detectada (techo, paredes, suelo)
Puntos de colores variados: Características extraídas del entorno con información RGB
Densidad variable: Áreas con más puntos tienen mejor calidad de escaneo

En tus imágenes se observa:

Estructura del techo con vigas
Paredes del edificio
Objetos en el entorno (escritorios, pantallas de computadora)
El robot o sensor moviéndose por el espacio

2. Vista de Odometría (Panel Inferior Izquierdo)
Esta vista muestra la perspectiva de la cámara con overlays de información:
Códigos de Color explicados en Imagen 4:
🔴 Fondo Rojo Oscuro (Dark Red) = Odometry Lost

Indica áreas donde se perdió el seguimiento de la odometría
Problema crítico: el sistema no puede determinar su posición

🟡 Amarillo Oscuro (Dark Yellow) = Low Inliers

Pocas correspondencias de características entre frames
Señal de advertencia: el mapeo puede ser inestable

🟢 Verde = Inliers

Características correctamente emparejadas entre frames consecutivos
Indica buen seguimiento visual

🟡 Amarillo = Not matched features from previous frame(s)

Características visibles pero no emparejadas con frames anteriores
Normal en áreas nuevas del entorno

🔴 Rojo = Outliers

Correspondencias incorrectas o ruido
Se filtran para no contaminar el mapa

En tus imágenes 4 y 5:

El fondo completamente rojo indica que se perdió la odometría
Esto sucede típicamente por:

Movimiento muy rápido de la cámara
Superficies sin textura (paredes lisas)
Iluminación pobre
Oclusiones o desenfoque



3. Mapa 3D (Panel Derecho)
Muestra la representación tridimensional construida:

Malla del suelo: Superficie plana (piso del edificio)
Estructuras verticales: Paredes y columnas
Ejes de coordenadas (verde y azul): Sistema de referencia del mapa

Verde = eje Y
Azul = eje Z
(Rojo sería X, no visible en estas vistas)



Progresión del mapa:

Imagen 1: Vista amplia del entorno mapeado con el techo y múltiples estructuras
Imagen 2: Acercamiento al área de trabajo (escritorios, computadoras)
Imagen 3: Rotación de la vista mostrando diferentes perspectivas
Imágenes 4-5: Enfoque en área interior con pérdida de tracking

4. Loop Closure Detection (Panel Superior Izquierdo)
Este panel (visible en todas las imágenes) es crucial:

Detecta cuando el robot vuelve a un lugar ya visitado
Al reconocer la ubicación, corrige la deriva acumulada del mapa
Mejora la consistencia global del mapa
Es fundamental para mapeos de larga duración


🔍 Análisis de las Secuencias
Imágenes 1-3: Mapeo Exitoso

Nube de puntos densa y coherente
Vista de odometría limpia (sin colores de advertencia visibles)
Mapa 3D bien estructurado
El sistema está trackeando correctamente

Imágenes 4-5: Pérdida de Tracking

Fondo rojo completo en la vista de odometría
Indica pérdida total de la odometría visual
El sistema no puede determinar su posición
Causas probables:

Superficie sin características visuales distintivas (pared lisa)
Movimiento brusco
Reflexiones en las pantallas de computadora
Cambio drástico de iluminación




📈 Calidad del Mapeo
Aspectos Positivos:

Captura detallada del techo y estructuras superiores
Buena definición de objetos (escritorios, computadoras)
Múltiples perspectivas del entorno

Áreas de Mejora:

Pérdida de tracking en las últimas imágenes
Algunas áreas con densidad de puntos irregular
Necesidad de movimientos más lentos y suaves
