#!/usr/bin/env python3
"""
Sensor Operations Script
========================
A unified automation tool for Ouster 3D LiDAR and Orbbec RGB-D Camera operations.

This script provides a CLI interface to:
- Detect and verify sensor connections
- Visualize sensor data streams
- Run SLAM operations (LiDAR)
- Record and playback data
- Convert/export point cloud data

Supported Sensors:
- Ouster 3D LiDAR (Ethernet connection)
- Orbbec RGB-D Camera (USB connection)

Requirements:
- Ouster LiDAR: ouster-sdk (pip install ouster-sdk)
- Orbbec Camera: pyorbbecsdk (pip install pyorbbecsdk)

Usage:
    python3 sensor_ops.py

Author: Andres Santiago Santafe Silva
License: MIT
"""

import subprocess
import sys
import re
import os
import time
from enum import Enum
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod


# ==============================================================================
# CONFIGURATION CONSTANTS
# ==============================================================================

class Config:
    """Central configuration for sensor operations."""
    
    # Network Configuration (Ouster LiDAR)
    ETHERNET_INTERFACE = "eno1"
    COMPUTER_IP = "169.254.41.100"
    CIDR = "16"
    LIDAR_DEFAULT_IP = "169.254.41.35"
    
    # UDP Ports for Ouster
    OUSTER_LIDAR_PORT = 7502
    OUSTER_IMU_PORT = 7503
    
    # Default recording settings
    DEFAULT_MAP_RATIO = "0.05"
    DEFAULT_RECORDING_DIR = os.path.expanduser("~/sensor_recordings")


class SensorType(Enum):
    """Enumeration of supported sensor types."""
    LIDAR = "lidar"
    CAMERA = "camera"


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width)


def print_step(step_num: int, description: str) -> None:
    """Print a formatted step indicator."""
    print(f"\n--- Step {step_num}: {description} ---")


def print_success(message: str) -> None:
    """Print a success message with green indicator."""
    print(f"[✓] {message}")


def print_error(message: str) -> None:
    """Print an error message with red indicator."""
    print(f"[✗] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"[!] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"[i] {message}")


def run_command(
    command: List[str] | str,
    shell: bool = False,
    capture_output: bool = True,
    check: bool = True,
    timeout: Optional[int] = None
) -> Optional[str]:
    """
    Execute a shell command with error handling.
    
    Args:
        command: Command to execute (list or string)
        shell: Whether to run through shell
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise exception on non-zero exit
        timeout: Command timeout in seconds
    
    Returns:
        Command stdout if successful, None if failed
    """
    try:
        cmd_str = ' '.join(command) if isinstance(command, list) else command
        print(f"[>] Running: {cmd_str}")
        
        result = subprocess.run(
            command,
            shell=shell,
            check=check,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {e.stderr if e.stderr else str(e)}")
        return None
    except subprocess.TimeoutExpired:
        print_error(f"Command timed out after {timeout} seconds")
        return None
    except FileNotFoundError:
        print_error(f"Command not found: {command[0] if isinstance(command, list) else command.split()[0]}")
        return None


def check_sudo() -> bool:
    """
    Verify if script is running with root privileges.
    
    Returns:
        True if running as root, False otherwise
    """
    if os.geteuid() != 0:
        print_error("This operation requires root privileges.")
        print_info("Please run with: sudo python3 sensor_ops.py")
        return False
    return True


def detect_virtual_environment() -> str:
    """
    Auto-detect the active virtual environment bin directory.
    
    Searches for common virtual environment names in the current
    working directory and returns the bin path.
    
    Returns:
        Path to the virtual environment bin directory
    """
    cwd = os.getcwd()
    venv_names = [".venv", "ouster_env", "orbbec_env", "venv", "env"]
    
    for venv in venv_names:
        bin_path = os.path.join(cwd, venv, "bin")
        if os.path.exists(os.path.join(bin_path, "python3")):
            print_info(f"Detected virtual environment: {bin_path}")
            return bin_path
    
    # Fallback to system bin
    return os.path.dirname(sys.executable)


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# ==============================================================================
# ABSTRACT SENSOR BASE CLASS
# ==============================================================================

class SensorBase(ABC):
    """Abstract base class for sensor operations."""
    
    def __init__(self):
        self.venv_bin = detect_virtual_environment()
    
    @abstractmethod
    def verify_connection(self) -> bool:
        """Verify sensor connection. Returns True if connected."""
        pass
    
    @abstractmethod
    def discover(self) -> Optional[str]:
        """Discover sensor and return identifier (IP or device ID)."""
        pass
    
    @abstractmethod
    def visualize(self) -> None:
        """Launch visualization interface."""
        pass
    
    @abstractmethod
    def get_menu_options(self) -> List[Tuple[str, str]]:
        """Return list of (option_key, description) tuples for menu."""
        pass
    
    @abstractmethod
    def handle_option(self, option: str) -> None:
        """Handle a menu option selection."""
        pass


# ==============================================================================
# OUSTER LIDAR IMPLEMENTATION (UPDATED)
# ==============================================================================

class OusterLiDAR(SensorBase):
    """
    Ouster 3D LiDAR sensor operations.
    """
    
    def __init__(self):
        super().__init__()
        self.ouster_exec = os.path.join(self.venv_bin, "ouster-cli")
        self.sensor_ip: Optional[str] = None
    
    def _check_ouster_cli(self) -> bool:
        """Verify ouster-cli is installed and accessible."""
        if os.path.exists(self.ouster_exec):
            return True
        
        # Try to find in PATH
        result = run_command(["which", "ouster-cli"], check=False)
        if result:
            self.ouster_exec = result
            return True
        
        print_error(f"ouster-cli not found at {self.ouster_exec}")
        print_info("Install with: pip install ouster-sdk")
        return False
    
    def verify_connection(self) -> bool:
        """
        Verify LiDAR network connection and AUTO-CONFIGURE IP if missing.
        """
        print_step(1, "Verifying LiDAR Connection")
        
        # --- NEW AUTOMATIC IP FIX START ---
        # Check if eno1 has the correct IP assigned
        print_info(f"Checking configuration for interface: {Config.ETHERNET_INTERFACE}")
        ip_check = run_command(["ip", "addr", "show", Config.ETHERNET_INTERFACE], check=False)
        
        if ip_check and Config.COMPUTER_IP not in ip_check:
            print_warning(f"IP {Config.COMPUTER_IP} is missing on {Config.ETHERNET_INTERFACE}")
            print_info("Attempting auto-configuration (Sudo password may be required)...")
            
            # Auto-run the IP command with sudo
            run_command(["sudo", "ip", "addr", "add", 
                         f"{Config.COMPUTER_IP}/{Config.CIDR}", 
                         "dev", Config.ETHERNET_INTERFACE], check=False)
            run_command(["sudo", "ip", "link", "set", Config.ETHERNET_INTERFACE, "up"], check=False)
            
            # Quick check if it worked
            time.sleep(1)
        # --- NEW AUTOMATIC IP FIX END ---

        # First check if ouster-cli exists
        if not self._check_ouster_cli():
            return False
        
        # Try to ping the default sensor IP
        result = run_command(
            ["ping", "-c", "1", "-W", "2", Config.LIDAR_DEFAULT_IP],
            check=False,
            timeout=5
        )
        
        if result is not None:
            print_success(f"LiDAR reachable at {Config.LIDAR_DEFAULT_IP}")
            self.sensor_ip = Config.LIDAR_DEFAULT_IP
            return True
        
        print_warning(f"Cannot ping {Config.LIDAR_DEFAULT_IP}")
        return False
    
    def configure_network(self) -> bool:
        """
        Manual Network Configuration
        """
        print_header("Network Configuration for Ouster LiDAR")
        
        # We prepend sudo here so we don't need to run the whole script as root
        print_step(1, "Configuring IP Address")
        ip_cmd = ["sudo", "ip", "addr", "add", 
                  f"{Config.COMPUTER_IP}/{Config.CIDR}", 
                  "dev", Config.ETHERNET_INTERFACE]
        result = run_command(ip_cmd, check=False)
        
        # Bring interface up
        print_step(2, "Activating Network Interface")
        run_command(["sudo", "ip", "link", "set", Config.ETHERNET_INTERFACE, "up"])
        
        # Configure firewall
        print_step(3, "Configuring Firewall (UFW)")
        for port in [Config.OUSTER_LIDAR_PORT, Config.OUSTER_IMU_PORT]:
            run_command(["sudo", "ufw", "allow", f"{port}/udp"], check=False)
        
        print_success("Network configuration complete")
        return True
    
    def discover(self) -> Optional[str]:
        """Discover Ouster sensors on the network."""
        print_step(1, "Discovering Ouster Sensors")
        
        if not self._check_ouster_cli():
            return None
        
        output = run_command([self.ouster_exec, "discover"], timeout=15)
        
        if output:
            print_success("Discovery complete")
            print(output)
            
            # Extract IP addresses from output
            ips = re.findall(r'[0-9]+(?:\.[0-9]+){3}', output)
            if ips:
                self.sensor_ip = ips[0]
                print_success(f"Found sensor at: {self.sensor_ip}")
                return self.sensor_ip
        
        print_warning("No sensor found via discovery")
        print_info(f"Using default IP: {Config.LIDAR_DEFAULT_IP}")
        self.sensor_ip = Config.LIDAR_DEFAULT_IP
        return self.sensor_ip
    
    def visualize(self) -> None:
        """Launch the Ouster point cloud visualizer."""
        if not self.sensor_ip:
            self.discover()
        
        if not self.sensor_ip:
            print_error("No sensor IP available")
            return
        
        print_header(f"Launching Visualizer for {self.sensor_ip}")
        print_info("Press 'q' or close window to exit")
        
        subprocess.run([self.ouster_exec, "source", self.sensor_ip, "viz"])
    
    def run_slam(self, save_file: Optional[str] = None) -> None:
        """
        Run GLIM SLAM using ROS 2.
        Based on GLIM SLAM Quick Start Guide.
        """
        # Define paths based on your guide
        workspace_dir = "/home/cpsstudent/Desktop/CPSPERRO/my_ros2_ws"
        config_path = os.path.join(workspace_dir, "install/glim/share/glim/config")
        rviz_config = os.path.join(workspace_dir, "install/glim/share/glim/config/glim.rviz")

        print_header("Launching GLIM SLAM (ROS 2)")
        
        # Check if directories exist
        if not os.path.exists(workspace_dir):
            print_error(f"Workspace not found at: {workspace_dir}")
            return

        # Warning about data source
        print_warning("Ensure your Ouster ROS Driver is running and publishing topics!")
        print_info("GLIM expects: /ouster/imu and /ouster/points")

        try:
            # 1. Command for GLIM Node
            glim_cmd = [
                "ros2", "run", "glim_ros", "glim_rosnode",
                "--ros-args", "-p", f"config_path:={config_path}"
            ]

            # 2. Command for RViz Visualization
            rviz_cmd = [
                "ros2", "run", "rviz2", "rviz2",
                "-d", rviz_config
            ]

            print_step(1, "Starting GLIM Node")
            print_info(f"Config Path: {config_path}")
            # Use Popen to run in background (non-blocking)
            glim_process = subprocess.Popen(glim_cmd, cwd=workspace_dir)
            
            # Wait a few seconds for GLIM to initialize
            print_info("Waiting for GLIM to initialize...")
            time.sleep(3)

            print_step(2, "Starting RViz Visualization")
            # Run RViz
            rviz_process = subprocess.Popen(rviz_cmd, cwd=workspace_dir)

            print_success("SLAM System Running")
            print_info("Press Ctrl+C to stop all processes and exit SLAM mode.")

            # Keep the script running to monitor processes
            while True:
                time.sleep(1)
                if glim_process.poll() is not None:
                    print_error("GLIM Node crashed or stopped unexpectedly.")
                    break

        except KeyboardInterrupt:
            print("\n")
            print_warning("Stopping SLAM processes...")
            
            # Terminate processes gracefully
            if 'rviz_process' in locals():
                rviz_process.terminate()
            if 'glim_process' in locals():
                glim_process.terminate()
            
            # Wait for them to close
            time.sleep(1)
            print_success("SLAM stopped.")
            
        except FileNotFoundError:
            print_error("ros2 command not found. Did you source your ROS 2 environment?")
            print_info("Run: source /opt/ros/jazzy/setup.bash (or your distro)")
    
    def convert_map(self, input_file: str, output_file: str) -> None:
        """Convert OSF map file to PLY point cloud format."""
        print_header("Converting Map")
        
        if not os.path.exists(input_file):
            print_error(f"Input file not found: {input_file}")
            return
        
        if not output_file.endswith('.ply'):
            output_file += '.ply'
        
        subprocess.run([
            self.ouster_exec, "source", input_file, "convert", output_file
        ])
        
        if os.path.exists(output_file):
            print_success(f"Conversion complete: {output_file}")
    
    def get_menu_options(self) -> List[Tuple[str, str]]:
        """Return LiDAR-specific menu options."""
        return [
            ("1", "Manual Network Config (Force Sudo)"),
            ("2", "Discover & Visualize"),
            ("3", "Discover & Run SLAM"),
            ("4", "Run SLAM & Save Map"),
            ("5", "Convert Map (.osf → .ply)"),
            ("0", "Back to Main Menu"),
        ]
    
    def handle_option(self, option: str) -> None:
        """Handle LiDAR menu option selection."""
        if option == "1":
            self.configure_network()
        elif option == "2":
            self.discover()
            self.visualize()
        elif option == "3":
            self.discover()
            self.run_slam()
        elif option == "4":
            self.discover()
            filename = input("Enter filename for map (e.g., my_map.osf): ").strip()
            if filename:
                self.run_slam(save_file=filename)
        elif option == "5":
            input_file = input("Input .osf file path: ").strip()
            output_file = input("Output .ply file path: ").strip()
            if input_file and output_file:
                self.convert_map(input_file, output_file)

# ==============================================================================
# ORBBEC CAMERA IMPLEMENTATION
# ==============================================================================

class OrbbecCamera(SensorBase):
    """
    Orbbec RGB-D Camera sensor operations.
    
    Handles USB connection verification, device discovery,
    stream visualization, recording, and point cloud export.
    """
    
    def __init__(self):
        super().__init__()
        self.device_info: Optional[dict] = None
        self._sdk_available = False
    
    def _check_sdk(self) -> bool:
        """Verify pyorbbecsdk is installed."""
        try:
            import pyorbbecsdk
            self._sdk_available = True
            print_success(f"pyorbbecsdk version available")
            return True
        except ImportError:
            print_error("pyorbbecsdk not found")
            print_info("Install with: pip install pyorbbecsdk")
            print_info("Or build from source: https://github.com/orbbec/pyorbbecsdk")
            return False
    
    def _check_udev_rules(self) -> bool:
        """Check if udev rules are installed for USB access."""
        udev_rule_path = "/etc/udev/rules.d/99-obsensor-libusb.rules"
        if os.path.exists(udev_rule_path):
            return True
        
        # Check alternative rule file
        alt_path = "/etc/udev/rules.d/99-orbbec.rules"
        return os.path.exists(alt_path)
    
    def verify_connection(self) -> bool:
        """
        Verify camera USB connection.
        
        Checks for Orbbec devices in USB device list and
        verifies SDK availability.
        """
        print_step(1, "Verifying Camera Connection")
        
        # Check SDK first
        if not self._check_sdk():
            return False
        
        # Check USB devices for Orbbec (VID: 2bc5)
        result = run_command(["lsusb"], check=False)
        if result and "2bc5" in result.lower():
            print_success("Orbbec camera detected on USB")
            
            # Check udev rules
            if not self._check_udev_rules():
                print_warning("udev rules may not be installed")
                print_info("Run: sudo bash ./install_udev_rules.sh from pyorbbecsdk/scripts")
            
            return True
        
        print_error("No Orbbec camera detected on USB")
        print_info("Please ensure the camera is connected via USB")
        return False
    
    def discover(self) -> Optional[str]:
        """
        Discover connected Orbbec cameras.
        
        Uses pyorbbecsdk to enumerate connected devices
        and returns device information.
        """
        print_step(1, "Discovering Orbbec Cameras")
        
        if not self._check_sdk():
            return None
        
        try:
            from pyorbbecsdk import Context
            
            ctx = Context()
            device_list = ctx.query_devices()
            device_count = device_list.get_device_count()
            
            if device_count == 0:
                print_warning("No Orbbec devices found")
                print_info("Ensure camera is connected and udev rules are installed")
                return None
            
            print_success(f"Found {device_count} device(s)")
            
            # Get first device info
            device = device_list.get_device(0)
            device_info = device.get_device_info()
            
            self.device_info = {
                "name": device_info.get_name(),
                "pid": device_info.get_pid(),
                "vid": device_info.get_vid(),
                "serial": device_info.get_serial_number(),
                "firmware": device_info.get_firmware_version(),
            }
            
            print(f"  Device Name:     {self.device_info['name']}")
            print(f"  Serial Number:   {self.device_info['serial']}")
            print(f"  Firmware:        {self.device_info['firmware']}")
            
            return self.device_info['serial']
            
        except Exception as e:
            print_error(f"Discovery failed: {e}")
            return None
    
    def visualize(self) -> None:
        """Launch combined depth and color stream visualization."""
        print_header("Launching Camera Visualizer")
        self._run_viewer("combined")
    
    def visualize_depth(self) -> None:
        """Launch depth stream visualization only."""
        print_header("Launching Depth Viewer")
        self._run_viewer("depth")
    
    def visualize_color(self) -> None:
        """Launch color stream visualization only."""
        print_header("Launching Color Viewer")
        self._run_viewer("color")
    
    def visualize_point_cloud(self) -> None:
        """Launch 3D point cloud visualization."""
        print_header("Launching Point Cloud Viewer")
        print_warning("Requires Open3D: pip install open3d")
        self._run_viewer("pointcloud")
    
    def _run_viewer(self, mode: str) -> None:
        """
        Internal method to run different visualization modes.
        
        Args:
            mode: One of 'depth', 'color', 'combined', 'pointcloud'
        """
        if not self._check_sdk():
            return
        
        print_info("Press 'q' or ESC to exit viewer")
        
        viewer_script = self._generate_viewer_script(mode)
        
        # Execute the viewer script
        try:
            exec(viewer_script, {"__name__": "__main__"})
        except KeyboardInterrupt:
            print_info("\nViewer closed")
        except Exception as e:
            print_error(f"Viewer error: {e}")
    
    def _generate_viewer_script(self, mode: str) -> str:
        """Generate viewer script based on mode."""
        
        if mode == "depth":
            return '''
import cv2
import numpy as np
from pyorbbecsdk import Config, Pipeline, OBSensorType, OBFormat

config = Config()
pipeline = Pipeline()

try:
    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
except Exception as e:
    print(f"Error configuring depth stream: {e}")
    raise

pipeline.start(config)
print("Depth viewer started. Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            continue
        
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        data = data.reshape((height, width))
        
        # Normalize for visualization
        data_normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        data_colored = cv2.applyColorMap(data_normalized, cv2.COLORMAP_JET)
        
        cv2.imshow("Orbbec Depth", data_colored)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
'''
        
        elif mode == "color":
            return '''
import cv2
import numpy as np
from pyorbbecsdk import Config, Pipeline, OBSensorType, OBFormat

config = Config()
pipeline = Pipeline()

try:
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)
except Exception as e:
    print(f"Error configuring color stream: {e}")
    raise

pipeline.start(config)
print("Color viewer started. Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        
        width = color_frame.get_width()
        height = color_frame.get_height()
        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        
        # Handle different formats
        fmt = color_frame.get_format()
        if data.size == width * height * 3:
            image = data.reshape((height, width, 3))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            continue
        
        cv2.imshow("Orbbec Color", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
'''
        
        elif mode == "combined":
            return '''
import cv2
import numpy as np
from pyorbbecsdk import Config, Pipeline, OBSensorType

config = Config()
pipeline = Pipeline()

# Configure depth stream
try:
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
except:
    print("Warning: Could not configure depth stream")

# Configure color stream
try:
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)
except:
    print("Warning: Could not configure color stream")

pipeline.start(config)
print("Combined viewer started. Press 'q' to quit.")

try:
    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        
        # Process depth frame
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            data = data.reshape((height, width))
            data_norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(data_norm, cv2.COLORMAP_JET)
            cv2.imshow("Orbbec Depth", depth_colored)
        
        # Process color frame
        color_frame = frames.get_color_frame()
        if color_frame:
            width = color_frame.get_width()
            height = color_frame.get_height()
            data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            if data.size == width * height * 3:
                image = data.reshape((height, width, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Orbbec Color", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
'''
        
        elif mode == "pointcloud":
            return '''
import numpy as np
from pyorbbecsdk import Config, Pipeline, OBSensorType, OBAlignMode

try:
    import open3d as o3d
except ImportError:
    print("Open3D required: pip install open3d")
    raise

config = Config()
pipeline = Pipeline()

# Configure streams with depth-to-color alignment
depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
depth_profile = depth_profiles.get_default_video_stream_profile()
config.enable_stream(depth_profile)

color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
color_profile = color_profiles.get_default_video_stream_profile()
config.enable_stream(color_profile)

config.set_align_mode(OBAlignMode.SW_MODE)

pipeline.start(config)
print("Point cloud viewer started. Close window to quit.")

vis = o3d.visualization.Visualizer()
vis.create_window("Orbbec Point Cloud")
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

try:
    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if depth_frame is None:
            continue
        
        # Get point cloud from SDK
        points = frames.get_point_cloud(pipeline.get_camera_param())
        if points is not None and len(points) > 0:
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            
            if color_frame and points.shape[1] >= 6:
                colors = points[:, 3:6] / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            vis.update_geometry(pcd)
        
        if not vis.poll_events():
            break
        vis.update_renderer()
finally:
    pipeline.stop()
    vis.destroy_window()
'''
        
        return ""
    
    def record_streams(self, filename: str, duration: int = 30) -> None:
        """
        Record camera streams to file.
        
        Args:
            filename: Output filename (without extension)
            duration: Recording duration in seconds
        """
        print_header("Recording Camera Streams")
        
        if not self._check_sdk():
            return
        
        ensure_directory(Config.DEFAULT_RECORDING_DIR)
        output_path = os.path.join(Config.DEFAULT_RECORDING_DIR, f"{filename}.bag")
        
        print_info(f"Recording to: {output_path}")
        print_info(f"Duration: {duration} seconds")
        print_info("Press Ctrl+C to stop early")
        
        try:
            from pyorbbecsdk import Config, Pipeline, OBSensorType, Recorder
            
            config = Config()
            pipeline = Pipeline()
            
            # Enable streams
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            config.enable_stream(depth_profiles.get_default_video_stream_profile())
            
            try:
                color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                config.enable_stream(color_profiles.get_default_video_stream_profile())
            except:
                print_warning("Color stream not available")
            
            pipeline.start(config)
            
            # Note: Recording implementation depends on SDK version
            # This is a simplified example
            print_warning("Recording feature requires Orbbec Viewer or SDK recording API")
            print_info("Consider using: OrbbecViewer for full recording capabilities")
            
            pipeline.stop()
            
        except Exception as e:
            print_error(f"Recording failed: {e}")
    
    def save_point_cloud(self, filename: str) -> None:
        """
        Capture and save a single point cloud frame.
        
        Args:
            filename: Output filename (without extension)
        """
        print_header("Saving Point Cloud")
        
        if not self._check_sdk():
            return
        
        try:
            import open3d as o3d
        except ImportError:
            print_error("Open3D required: pip install open3d")
            return
        
        ensure_directory(Config.DEFAULT_RECORDING_DIR)
        output_path = os.path.join(Config.DEFAULT_RECORDING_DIR, f"{filename}.ply")
        
        print_info(f"Saving to: {output_path}")
        print_info("Capturing point cloud...")
        
        try:
            from pyorbbecsdk import Config, Pipeline, OBSensorType
            
            config = Config()
            pipeline = Pipeline()
            
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            config.enable_stream(depth_profiles.get_default_video_stream_profile())
            
            pipeline.start(config)
            
            # Capture a few frames to let auto-exposure settle
            for _ in range(10):
                frames = pipeline.wait_for_frames(100)
            
            frames = pipeline.wait_for_frames(1000)
            if frames:
                points = frames.get_point_cloud(pipeline.get_camera_param())
                
                if points is not None and len(points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
                    
                    o3d.io.write_point_cloud(output_path, pcd)
                    print_success(f"Point cloud saved: {output_path}")
                else:
                    print_error("Failed to generate point cloud")
            
            pipeline.stop()
            
        except Exception as e:
            print_error(f"Save failed: {e}")
    
    def get_menu_options(self) -> List[Tuple[str, str]]:
        """Return camera-specific menu options."""
        return [
            ("1", "Discover & Get Device Info"),
            ("2", "View Depth Stream"),
            ("3", "View Color Stream"),
            ("4", "View Combined Streams"),
            ("5", "View 3D Point Cloud"),
            ("6", "Save Point Cloud (.ply)"),
            ("7", "Record Streams"),
            ("0", "Back to Main Menu"),
        ]
    
    def handle_option(self, option: str) -> None:
        """Handle camera menu option selection."""
        if option == "1":
            self.discover()
        elif option == "2":
            self.visualize_depth()
        elif option == "3":
            self.visualize_color()
        elif option == "4":
            self.visualize()
        elif option == "5":
            self.visualize_point_cloud()
        elif option == "6":
            filename = input("Enter filename for point cloud: ").strip()
            if filename:
                self.save_point_cloud(filename)
            else:
                print_warning("No filename provided")
        elif option == "7":
            filename = input("Enter recording filename: ").strip()
            duration = input("Recording duration (seconds, default 30): ").strip()
            duration = int(duration) if duration.isdigit() else 30
            if filename:
                self.record_streams(filename, duration)
            else:
                print_warning("No filename provided")


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class SensorOpsApp:
    """Main application controller for sensor operations."""
    
    def __init__(self):
        self.lidar = OusterLiDAR()
        self.camera = OrbbecCamera()
        self.current_sensor: Optional[SensorBase] = None
    
    def print_main_menu(self) -> None:
        """Display the main sensor selection menu."""
        print_header("Sensor Operations Tool")
        print("\nSelect a sensor to work with:")
        print("  [1] Ouster 3D LiDAR (Ethernet)")
        print("  [2] Orbbec RGB-D Camera (USB)")
        print("  [0] Exit")
    
    def print_sensor_menu(self) -> None:
        """Display the current sensor's operation menu."""
        if not self.current_sensor:
            return
        
        sensor_name = "LiDAR" if isinstance(self.current_sensor, OusterLiDAR) else "Camera"
        print_header(f"{sensor_name} Operations")
        
        print("\nAvailable options:")
        for key, description in self.current_sensor.get_menu_options():
            print(f"  [{key}] {description}")
    
    def run_sensor_menu(self) -> None:
        """Run the sensor-specific operation menu loop."""
        while True:
            self.print_sensor_menu()
            
            choice = input("\nSelect option: ").strip()
            
            if choice == "0":
                break
            
            self.current_sensor.handle_option(choice)
            
            input("\nPress Enter to continue...")
    
    def run(self) -> None:
        """Run the main application loop."""
        print("\n" + "=" * 60)
        print(" SENSOR OPERATIONS AUTOMATION TOOL ".center(60))
        print(" Ouster LiDAR & Orbbec Camera ".center(60))
        print("=" * 60)
        
        while True:
            self.print_main_menu()
            
            choice = input("\nSelect sensor [0-2]: ").strip()
            
            if choice == "0":
                print_info("Exiting. Goodbye!")
                break
            elif choice == "1":
                self.current_sensor = self.lidar
                print_info("Selected: Ouster LiDAR")
                if self.lidar.verify_connection():
                    self.run_sensor_menu()
                else:
                    print_warning("Connection verification failed")
                    retry = input("Continue anyway? [y/N]: ").strip().lower()
                    if retry == 'y':
                        self.run_sensor_menu()
            elif choice == "2":
                self.current_sensor = self.camera
                print_info("Selected: Orbbec Camera")
                if self.camera.verify_connection():
                    self.run_sensor_menu()
                else:
                    print_warning("Connection verification failed")
                    retry = input("Continue anyway? [y/N]: ").strip().lower()
                    if retry == 'y':
                        self.run_sensor_menu()
            else:
                print_warning("Invalid selection. Please enter 0, 1, or 2.")
            
            print()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    """Application entry point."""
    app = SensorOpsApp()
    
    try:
        app.run()
    except KeyboardInterrupt:
        print_info("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
