#!/usr/bin/env python3
"""
Sensor Operations Script
========================
A unified automation tool for Ouster 3D LiDAR and Orbbec RGB-D Camera operations.

This script provides a CLI interface to:
- Detect and verify sensor connections
- Visualize sensor data streams
- Launch ROS 2 Ouster driver
- Run GLIM SLAM with proper initialization
- Record and playback data
- Convert/export point cloud data

Supported Sensors:
- Ouster 3D LiDAR (Ethernet connection)
- Orbbec RGB-D Camera (USB connection)

Configuration:
- Edit sensor_ops_config.yaml for persistent settings
- Use command line arguments to override settings

Usage:
    python3 sensor_ops.py
    python3 sensor_ops.py --config /path/to/config.yaml
    python3 sensor_ops.py --lidar-ip 169.254.41.35 --ros-ws /path/to/ws

Author: Andres Santiago Santafe Silva
License: MIT
"""

import subprocess
import sys
import re
import os
import time
import argparse
import signal
from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# Try to import YAML parser
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ==============================================================================
# CONFIGURATION MANAGEMENT
# ==============================================================================

@dataclass
class OusterConfig:
    """Ouster LiDAR configuration."""
    ethernet_interface: str = "eno1"
    computer_ip: str = "169.254.41.100"
    cidr: str = "16"
    sensor_ip: str = "169.254.41.35"
    lidar_port: int = 7502
    imu_port: int = 7503


@dataclass
class ROS2Config:
    """ROS 2 configuration."""
    workspace: str = "/home/cpsstudent/Desktop/CPSPERRO/my_ros2_ws"
    ros_setup: str = "/opt/ros/jazzy/setup.bash"
    ouster_params_file: str = "ouster_params.yaml"


@dataclass
class GLIMConfig:
    """GLIM SLAM configuration."""
    config_path: str = "install/glim/share/glim/config"
    rviz_config: str = "install/glim/share/glim/config/glim.rviz"
    pointcloud_topic: str = "/ouster/points"
    imu_topic: str = "/ouster/imu"


@dataclass
class OrbbecConfig:
    """Orbbec camera configuration."""
    vendor_id: str = "2bc5"


@dataclass
class GeneralConfig:
    """General settings."""
    recording_dir: str = "~/sensor_recordings"
    default_map_ratio: str = "0.05"


@dataclass
class AppConfig:
    """Main application configuration container."""
    ouster: OusterConfig = field(default_factory=OusterConfig)
    ros2: ROS2Config = field(default_factory=ROS2Config)
    glim: GLIMConfig = field(default_factory=GLIMConfig)
    orbbec: OrbbecConfig = field(default_factory=OrbbecConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'AppConfig':
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            print_warning("PyYAML not installed. Using default configuration.")
            print_info("Install with: pip install pyyaml")
            return cls()
        
        filepath = Path(filepath).expanduser()
        if not filepath.exists():
            print_warning(f"Config file not found: {filepath}")
            return cls()
        
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            config = cls()
            
            # Parse ouster section
            if 'ouster' in data:
                o = data['ouster']
                config.ouster = OusterConfig(
                    ethernet_interface=o.get('ethernet_interface', config.ouster.ethernet_interface),
                    computer_ip=o.get('computer_ip', config.ouster.computer_ip),
                    cidr=str(o.get('cidr', config.ouster.cidr)),
                    sensor_ip=o.get('sensor_ip', config.ouster.sensor_ip),
                    lidar_port=o.get('lidar_port', config.ouster.lidar_port),
                    imu_port=o.get('imu_port', config.ouster.imu_port),
                )
            
            # Parse ros2 section
            if 'ros2' in data:
                r = data['ros2']
                config.ros2 = ROS2Config(
                    workspace=r.get('workspace', config.ros2.workspace),
                    ros_setup=r.get('ros_setup', config.ros2.ros_setup),
                    ouster_params_file=r.get('ouster_params_file', config.ros2.ouster_params_file),
                )
            
            # Parse glim section
            if 'glim' in data:
                g = data['glim']
                topics = g.get('topics', {})
                config.glim = GLIMConfig(
                    config_path=g.get('config_path', config.glim.config_path),
                    rviz_config=g.get('rviz_config', config.glim.rviz_config),
                    pointcloud_topic=topics.get('pointcloud', config.glim.pointcloud_topic),
                    imu_topic=topics.get('imu', config.glim.imu_topic),
                )
            
            # Parse orbbec section
            if 'orbbec' in data:
                config.orbbec = OrbbecConfig(
                    vendor_id=data['orbbec'].get('vendor_id', config.orbbec.vendor_id),
                )
            
            # Parse general section
            if 'general' in data:
                g = data['general']
                config.general = GeneralConfig(
                    recording_dir=g.get('recording_dir', config.general.recording_dir),
                    default_map_ratio=str(g.get('default_map_ratio', config.general.default_map_ratio)),
                )
            
            print_success(f"Configuration loaded from: {filepath}")
            return config
            
        except Exception as e:
            print_error(f"Failed to load config: {e}")
            return cls()
    
    def apply_cli_overrides(self, args: argparse.Namespace) -> None:
        """Apply command line argument overrides."""
        if args.lidar_ip:
            self.ouster.sensor_ip = args.lidar_ip
            print_info(f"Override: lidar_ip = {args.lidar_ip}")
        
        if args.computer_ip:
            self.ouster.computer_ip = args.computer_ip
            print_info(f"Override: computer_ip = {args.computer_ip}")
        
        if args.interface:
            self.ouster.ethernet_interface = args.interface
            print_info(f"Override: interface = {args.interface}")
        
        if args.ros_ws:
            self.ros2.workspace = args.ros_ws
            print_info(f"Override: ros_workspace = {args.ros_ws}")
        
        if args.ros_setup:
            self.ros2.ros_setup = args.ros_setup
            print_info(f"Override: ros_setup = {args.ros_setup}")


# Global config instance
CONFIG: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global CONFIG
    if CONFIG is None:
        CONFIG = AppConfig()
    return CONFIG


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
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """
    Execute a shell command with error handling.
    
    Args:
        command: Command to execute (list or string)
        shell: Whether to run through shell
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise exception on non-zero exit
        timeout: Command timeout in seconds
        cwd: Working directory
        env: Environment variables
    
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
            timeout=timeout,
            cwd=cwd,
            env=env
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


def run_bash_script(script: str, cwd: Optional[str] = None) -> Optional[subprocess.Popen]:
    """
    Run a bash script that sources ROS 2 environment.
    
    Args:
        script: Bash script content
        cwd: Working directory
    
    Returns:
        Popen process object
    """
    config = get_config()
    
    # Prepend ROS 2 sourcing
    full_script = f"""
#!/bin/bash
set -e
source {config.ros2.ros_setup}
if [ -f "{config.ros2.workspace}/install/setup.bash" ]; then
    source {config.ros2.workspace}/install/setup.bash
fi
{script}
"""
    
    try:
        process = subprocess.Popen(
            ['bash', '-c', full_script],
            cwd=cwd or config.ros2.workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        return process
    except Exception as e:
        print_error(f"Failed to run bash script: {e}")
        return None


def detect_virtual_environment() -> str:
    """
    Auto-detect the active virtual environment bin directory.
    
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
    
    return os.path.dirname(sys.executable)


def ensure_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    expanded = os.path.expanduser(path)
    os.makedirs(expanded, exist_ok=True)


def resolve_path(path: str, base: Optional[str] = None) -> str:
    """
    Resolve a path that may be relative or contain ~.
    
    Args:
        path: Path to resolve
        base: Base directory for relative paths
    
    Returns:
        Absolute resolved path
    """
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    if base:
        return os.path.join(base, expanded)
    return os.path.abspath(expanded)


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
# OUSTER LIDAR IMPLEMENTATION
# ==============================================================================

class OusterLiDAR(SensorBase):
    """
    Ouster 3D LiDAR sensor operations.
    
    Handles network configuration, discovery, visualization,
    ROS 2 driver launch, and GLIM SLAM operations.
    """
    
    def __init__(self):
        super().__init__()
        self.ouster_exec = os.path.join(self.venv_bin, "ouster-cli")
        self.sensor_ip: Optional[str] = None
        self._running_processes: List[subprocess.Popen] = []
    
    def _get_config(self) -> OusterConfig:
        """Get Ouster configuration."""
        return get_config().ouster
    
    def _check_ouster_cli(self) -> bool:
        """Verify ouster-cli is installed and accessible."""
        if os.path.exists(self.ouster_exec):
            return True
        
        result = run_command(["which", "ouster-cli"], check=False)
        if result:
            self.ouster_exec = result
            return True
        
        print_error(f"ouster-cli not found at {self.ouster_exec}")
        print_info("Install with: pip install ouster-sdk")
        return False
    
    def _check_ros2_environment(self) -> bool:
        """Check if ROS 2 environment is properly configured."""
        config = get_config()
        
        # Check ROS setup file
        if not os.path.exists(config.ros2.ros_setup):
            print_error(f"ROS 2 setup file not found: {config.ros2.ros_setup}")
            return False
        
        # Check workspace
        ws_path = Path(config.ros2.workspace)
        if not ws_path.exists():
            print_error(f"ROS 2 workspace not found: {config.ros2.workspace}")
            return False
        
        # Check workspace install directory
        install_path = ws_path / "install"
        if not install_path.exists():
            print_warning(f"Workspace not built: {install_path} not found")
            print_info("Run: cd {workspace} && colcon build")
            return False
        
        return True
    
    def verify_connection(self) -> bool:
        """
        Verify LiDAR network connection and auto-configure IP if missing.
        """
        print_step(1, "Verifying LiDAR Connection")
        
        cfg = self._get_config()
        
        # Check network interface configuration
        print_info(f"Checking interface: {cfg.ethernet_interface}")
        ip_check = run_command(["ip", "addr", "show", cfg.ethernet_interface], check=False)
        
        if ip_check and cfg.computer_ip not in ip_check:
            print_warning(f"IP {cfg.computer_ip} not assigned to {cfg.ethernet_interface}")
            print_info("Attempting auto-configuration...")
            
            run_command([
                "sudo", "ip", "addr", "add",
                f"{cfg.computer_ip}/{cfg.cidr}",
                "dev", cfg.ethernet_interface
            ], check=False)
            run_command([
                "sudo", "ip", "link", "set", cfg.ethernet_interface, "up"
            ], check=False)
            
            time.sleep(1)
        
        # Check ouster-cli
        if not self._check_ouster_cli():
            return False
        
        # Ping sensor
        result = run_command(
            ["ping", "-c", "1", "-W", "2", cfg.sensor_ip],
            check=False,
            timeout=5
        )
        
        if result is not None:
            print_success(f"LiDAR reachable at {cfg.sensor_ip}")
            self.sensor_ip = cfg.sensor_ip
            return True
        
        print_warning(f"Cannot ping {cfg.sensor_ip}")
        return False
    
    def configure_network(self) -> bool:
        """Manual network configuration for LiDAR."""
        print_header("Network Configuration for Ouster LiDAR")
        
        cfg = self._get_config()
        
        print_step(1, "Configuring IP Address")
        run_command([
            "sudo", "ip", "addr", "add",
            f"{cfg.computer_ip}/{cfg.cidr}",
            "dev", cfg.ethernet_interface
        ], check=False)
        
        print_step(2, "Activating Network Interface")
        run_command(["sudo", "ip", "link", "set", cfg.ethernet_interface, "up"])
        
        print_step(3, "Configuring Firewall (UFW)")
        for port in [cfg.lidar_port, cfg.imu_port]:
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
            
            ips = re.findall(r'[0-9]+(?:\.[0-9]+){3}', output)
            if ips:
                self.sensor_ip = ips[0]
                print_success(f"Found sensor at: {self.sensor_ip}")
                return self.sensor_ip
        
        cfg = self._get_config()
        print_warning("No sensor found via discovery")
        print_info(f"Using configured IP: {cfg.sensor_ip}")
        self.sensor_ip = cfg.sensor_ip
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
    
    def generate_ouster_params_yaml(self) -> str:
        """
        Generate the ouster_params.yaml content.
        
        Format matches what os_driver expects when launched with:
        ros2 run ouster_ros os_driver --ros-args --params-file ... -r __ns:=/ouster
        """
        cfg = self._get_config()
        
        # The node is remapped to /ouster namespace, so params go under /ouster/os_driver
        return f"""# Auto-generated Ouster ROS 2 Driver Parameters
# Generated by sensor_ops.py
# Usage: ros2 run ouster_ros os_driver --ros-args --params-file ouster_params.yaml -r __ns:=/ouster

/ouster/os_driver:
  ros__parameters:
    sensor_hostname: "{cfg.sensor_ip}"
    metadata_hostname: "{cfg.sensor_ip}"
    lidar_port: {cfg.lidar_port}
    imu_port: {cfg.imu_port}
    udp_dest_host: "{cfg.computer_ip}"
    auto_start: true
"""
    
    def launch_ouster_driver(self) -> Optional[subprocess.Popen]:
        """
        Launch the Ouster ROS 2 driver using ros2 run.
        
        Based on working command:
        ros2 run ouster_ros os_driver --ros-args --params-file ./ouster_params.yaml -r __ns:=/ouster
        
        Returns:
            Popen process object if successful
        """
        print_header("Launching Ouster ROS 2 Driver")
        
        if not self._check_ros2_environment():
            return None
        
        config = get_config()
        ws_path = config.ros2.workspace
        
        # Generate params file with correct format for os_driver node
        params_content = self.generate_ouster_params_yaml()
        params_file = os.path.join(ws_path, "ouster_params.yaml")
        
        print_step(1, "Writing parameters file")
        try:
            with open(params_file, 'w') as f:
                f.write(params_content)
            print_success(f"Parameters written to: {params_file}")
        except Exception as e:
            print_error(f"Failed to write params file: {e}")
            return None
        
        # Launch driver using ros2 run (the working method)
        print_step(2, "Starting Ouster Driver (os_driver node)")
        
        # Use the exact command format that works
        launch_script = f"""
ros2 run ouster_ros os_driver --ros-args --params-file {params_file} -r __ns:=/ouster
"""
        
        print_info("Command: ros2 run ouster_ros os_driver --ros-args --params-file ... -r __ns:=/ouster")
        
        process = run_bash_script(launch_script, cwd=ws_path)
        
        if process:
            self._running_processes.append(process)
            print_success("Ouster driver process started")
            print_info("Waiting for sensor initialization...")
            
            # Wait and check for initialization messages
            time.sleep(5)
            
            # Check if process is still running
            if process.poll() is None:
                print_success("Driver is running")
                print_info("Sensor should be publishing to /ouster/* topics")
                return process
            else:
                print_error("Driver exited unexpectedly")
                stdout, _ = process.communicate()
                if stdout:
                    print("Output:")
                    print(stdout[-2000:] if len(stdout) > 2000 else stdout)
                return None
        
        return None
    
    def verify_ros_topics(self) -> bool:
        """Verify that Ouster ROS topics are available."""
        print_step(1, "Verifying ROS 2 Topics")
        
        config = get_config()
        ws_path = config.ros2.workspace
        
        # Run ros2 topic list and capture output properly
        check_script = """
ros2 topic list 2>/dev/null
"""
        
        process = run_bash_script(check_script, cwd=ws_path)
        if process:
            try:
                stdout, _ = process.communicate(timeout=15)
                if stdout:
                    topics = stdout.strip().split('\n')
                    ouster_topics = [t for t in topics if 'ouster' in t.lower()]
                    
                    if ouster_topics:
                        print_success(f"Found {len(ouster_topics)} Ouster topic(s):")
                        for topic in ouster_topics:
                            print(f"  - {topic}")
                        
                        # Check for the essential topics
                        has_points = any('/points' in t for t in ouster_topics)
                        has_imu = any('/imu' in t for t in ouster_topics)
                        
                        if has_points:
                            print_success("Point cloud topic available")
                        if has_imu:
                            print_success("IMU topic available")
                        
                        return has_points  # At minimum we need points
                    else:
                        print_warning("No Ouster topics found in topic list")
                        print_info("Available topics:")
                        for topic in topics[:10]:  # Show first 10
                            print(f"  - {topic}")
                        if len(topics) > 10:
                            print(f"  ... and {len(topics) - 10} more")
            except subprocess.TimeoutExpired:
                print_error("Timeout waiting for topic list")
                process.kill()
        
        print_warning("Could not verify Ouster topics")
        print_info("The driver may still be initializing...")
        return False
    
    def run_glim_slam(self) -> None:
        """
        Run GLIM SLAM using the correct procedure.
        
        Based on the GLIM Quick Start Guide:
        1. Ensure Ouster driver is publishing topics
        2. Launch glim_rosnode with config_path
        3. Launch RViz for visualization
        """
        print_header("Launching GLIM SLAM")
        
        if not self._check_ros2_environment():
            return
        
        config = get_config()
        ws_path = config.ros2.workspace
        
        # Resolve GLIM paths
        glim_config_path = resolve_path(config.glim.config_path, ws_path)
        glim_rviz_config = resolve_path(config.glim.rviz_config, ws_path)
        
        # Verify paths exist
        if not os.path.exists(glim_config_path):
            print_error(f"GLIM config not found: {glim_config_path}")
            print_info("Ensure GLIM is built: colcon build --packages-select glim glim_ros")
            return
        
        print_info(f"GLIM Config Path: {glim_config_path}")
        print_info(f"RViz Config: {glim_rviz_config}")
        
        # Check if Ouster topics are available
        print_step(1, "Checking Ouster Topics")
        print_warning("Ensure Ouster ROS driver is running!")
        print_info(f"Expected topics: {config.glim.pointcloud_topic}, {config.glim.imu_topic}")
        
        proceed = input("\nIs the Ouster driver running? [y/N]: ").strip().lower()
        if proceed != 'y':
            print_info("Please launch Ouster driver first (Option 4)")
            return
        
        glim_process = None
        rviz_process = None
        
        try:
            # Launch GLIM Node
            print_step(2, "Starting GLIM Node")
            
            glim_script = f"""
ros2 run glim_ros glim_rosnode --ros-args -p config_path:={glim_config_path}
"""
            
            glim_process = run_bash_script(glim_script, cwd=ws_path)
            
            if not glim_process:
                print_error("Failed to start GLIM node")
                return
            
            self._running_processes.append(glim_process)
            print_success("GLIM node started")
            
            # Wait for GLIM initialization
            print_info("Waiting for GLIM to initialize (5 seconds)...")
            time.sleep(5)
            
            # Check if GLIM is still running
            if glim_process.poll() is not None:
                print_error("GLIM node crashed during initialization")
                stdout, _ = glim_process.communicate()
                if stdout:
                    print("Output:")
                    print(stdout[-2000:] if len(stdout) > 2000 else stdout)
                return
            
            # Launch RViz
            print_step(3, "Starting RViz Visualization")
            
            rviz_script = f"""
ros2 run rviz2 rviz2 -d {glim_rviz_config}
"""
            
            rviz_process = run_bash_script(rviz_script, cwd=ws_path)
            
            if rviz_process:
                self._running_processes.append(rviz_process)
                print_success("RViz started")
            
            print_success("GLIM SLAM System Running")
            print_info("Press Ctrl+C to stop all processes")
            
            # Monitor processes
            while True:
                time.sleep(1)
                
                # Check GLIM process
                if glim_process.poll() is not None:
                    print_error("GLIM node stopped unexpectedly")
                    break
                
                # Print any output (non-blocking)
                # This helps see GLIM's initialization messages
        
        except KeyboardInterrupt:
            print("\n")
            print_warning("Stopping SLAM processes...")
        
        finally:
            # Cleanup
            self._cleanup_processes([rviz_process, glim_process])
            print_success("SLAM stopped")
    
    def run_full_slam_pipeline(self) -> None:
        """
        Run the complete SLAM pipeline:
        1. Configure network
        2. Launch Ouster driver
        3. Wait for topics
        4. Launch GLIM SLAM
        """
        print_header("Full SLAM Pipeline")
        
        driver_process = None
        
        try:
            # Step 1: Verify/Configure Network
            print_step(1, "Network Configuration")
            if not self.verify_connection():
                print_warning("Network verification failed")
                proceed = input("Continue anyway? [y/N]: ").strip().lower()
                if proceed != 'y':
                    return
            
            # Step 2: Launch Ouster Driver
            print_step(2, "Launching Ouster Driver")
            driver_process = self.launch_ouster_driver()
            
            if not driver_process:
                print_error("Failed to launch Ouster driver")
                return
            
            # Wait for driver to fully initialize and show output
            print_info("Waiting for sensor to initialize and publish topics...")
            print_info("Driver output (watch for 'Sensor configured successfully'):")
            print("-" * 50)
            
            # Stream driver output for a while to see what's happening
            start_time = time.time()
            topics_found = False
            
            while time.time() - start_time < 20:  # 20 second window
                # Check if driver died
                if driver_process.poll() is not None:
                    print_error("Driver process exited!")
                    remaining_output, _ = driver_process.communicate()
                    if remaining_output:
                        print(remaining_output)
                    return
                
                # Try to read output (non-blocking)
                try:
                    import select
                    if select.select([driver_process.stdout], [], [], 0.1)[0]:
                        line = driver_process.stdout.readline()
                        if line:
                            print(f"  {line.rstrip()}")
                except:
                    time.sleep(0.5)
                
                # Periodically check for topics
                if time.time() - start_time > 10 and not topics_found:
                    if self._quick_topic_check():
                        topics_found = True
                        print("-" * 50)
                        print_success("Topics are now available!")
                        break
            
            print("-" * 50)
            
            # Step 3: Verify Topics
            print_step(3, "Verifying Topics")
            for attempt in range(3):
                if self.verify_ros_topics():
                    break
                print_info(f"Retry {attempt + 1}/3...")
                time.sleep(3)
            
            # Step 4: Launch GLIM
            print_step(4, "Launching GLIM SLAM")
            
            # Skip the confirmation since we just verified topics
            self._run_glim_without_confirmation()
        
        except KeyboardInterrupt:
            print("\n")
            print_warning("Pipeline interrupted")
        
        finally:
            if driver_process:
                print_info("Stopping driver...")
                self._cleanup_processes([driver_process])
    
    def _quick_topic_check(self) -> bool:
        """Quick check if ouster topics exist without full verification output."""
        config = get_config()
        ws_path = config.ros2.workspace
        
        check_script = "ros2 topic list 2>/dev/null | grep -q ouster && echo 'found'"
        process = run_bash_script(check_script, cwd=ws_path)
        if process:
            try:
                stdout, _ = process.communicate(timeout=5)
                return 'found' in stdout if stdout else False
            except:
                pass
        return False
    
    def _run_glim_without_confirmation(self) -> None:
        """Run GLIM SLAM without asking for confirmation (used in pipeline)."""
        config = get_config()
        ws_path = config.ros2.workspace
        
        glim_config_path = resolve_path(config.glim.config_path, ws_path)
        glim_rviz_config = resolve_path(config.glim.rviz_config, ws_path)
        
        if not os.path.exists(glim_config_path):
            print_error(f"GLIM config not found: {glim_config_path}")
            return
        
        print_info(f"GLIM Config Path: {glim_config_path}")
        
        glim_process = None
        rviz_process = None
        
        try:
            # Launch GLIM Node
            print_step(1, "Starting GLIM Node")
            
            glim_script = f"""
ros2 run glim_ros glim_rosnode --ros-args -p config_path:={glim_config_path}
"""
            
            glim_process = run_bash_script(glim_script, cwd=ws_path)
            
            if not glim_process:
                print_error("Failed to start GLIM node")
                return
            
            self._running_processes.append(glim_process)
            print_success("GLIM node started")
            
            print_info("Waiting for GLIM to initialize...")
            time.sleep(5)
            
            if glim_process.poll() is not None:
                print_error("GLIM node crashed during initialization")
                stdout, _ = glim_process.communicate()
                if stdout:
                    print("Output:")
                    print(stdout[-2000:] if len(stdout) > 2000 else stdout)
                return
            
            # Launch RViz
            print_step(2, "Starting RViz Visualization")
            
            rviz_script = f"""
ros2 run rviz2 rviz2 -d {glim_rviz_config}
"""
            
            rviz_process = run_bash_script(rviz_script, cwd=ws_path)
            
            if rviz_process:
                self._running_processes.append(rviz_process)
                print_success("RViz started")
            
            print_success("GLIM SLAM System Running")
            print_info("Press Ctrl+C to stop all processes")
            
            while True:
                time.sleep(1)
                if glim_process.poll() is not None:
                    print_error("GLIM node stopped unexpectedly")
                    break
        
        except KeyboardInterrupt:
            print("\n")
            print_warning("Stopping SLAM processes...")
        
        finally:
            self._cleanup_processes([rviz_process, glim_process])
    
    def _cleanup_processes(self, processes: List[Optional[subprocess.Popen]]) -> None:
        """Terminate running processes gracefully."""
        for proc in processes:
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
        
        # Remove from tracked processes
        for proc in processes:
            if proc in self._running_processes:
                self._running_processes.remove(proc)
    
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
            ("1", "Configure Network (Manual)"),
            ("2", "Discover Sensor"),
            ("3", "Visualize (ouster-cli viz)"),
            ("4", "Launch Ouster ROS 2 Driver (Background)"),
            ("5", "Launch Ouster Driver (Foreground - Debug)"),
            ("6", "Run GLIM SLAM (requires driver running)"),
            ("7", "Full Pipeline (Network → Driver → GLIM)"),
            ("8", "Convert Map (.osf → .ply)"),
            ("0", "Back to Main Menu"),
        ]
    
    def handle_option(self, option: str) -> None:
        """Handle LiDAR menu option selection."""
        if option == "1":
            self.configure_network()
        elif option == "2":
            self.discover()
        elif option == "3":
            self.discover()
            self.visualize()
        elif option == "4":
            self.discover()
            self.launch_ouster_driver()
            print_info("Driver running in background. Press Enter to stop...")
            input()
            self._cleanup_processes(self._running_processes.copy())
        elif option == "5":
            # Foreground mode for debugging
            self.discover()
            self.launch_ouster_driver_foreground()
        elif option == "6":
            self.run_glim_slam()
        elif option == "7":
            self.run_full_slam_pipeline()
        elif option == "8":
            input_file = input("Input .osf file path: ").strip()
            output_file = input("Output .ply file path: ").strip()
            if input_file and output_file:
                self.convert_map(input_file, output_file)
    
    def launch_ouster_driver_foreground(self) -> None:
        """
        Launch the Ouster driver in foreground mode for debugging.
        This blocks until Ctrl+C is pressed.
        """
        print_header("Launching Ouster Driver (Foreground Mode)")
        
        if not self._check_ros2_environment():
            return
        
        config = get_config()
        ws_path = config.ros2.workspace
        cfg = self._get_config()
        
        # Generate params file
        params_content = self.generate_ouster_params_yaml()
        params_file = os.path.join(ws_path, "ouster_params.yaml")
        
        try:
            with open(params_file, 'w') as f:
                f.write(params_content)
            print_success(f"Parameters written to: {params_file}")
        except Exception as e:
            print_error(f"Failed to write params file: {e}")
            return
        
        print_info("Starting driver in foreground. Press Ctrl+C to stop.")
        print_info("Watch the output to verify topics are being published.")
        print("-" * 60)
        
        # Build the full bash command
        full_script = f"""
#!/bin/bash
source {config.ros2.ros_setup}
source {ws_path}/install/setup.bash
ros2 run ouster_ros os_driver --ros-args --params-file {params_file} -r __ns:=/ouster
"""
        
        try:
            # Run in foreground (blocking)
            subprocess.run(['bash', '-c', full_script], cwd=ws_path)
        except KeyboardInterrupt:
            print("\n")
            print_info("Driver stopped by user")


# ==============================================================================
# ORBBEC CAMERA IMPLEMENTATION (COMPLETE WITH MENU METHODS)
# ==============================================================================

import os
import time
from typing import Optional, List, Tuple

class OrbbecCamera(SensorBase):
    """
    Orbbec RGB-D Camera sensor operations (Updated for pyorbbecsdk2).
    """
    
    def __init__(self):
        super().__init__()
        self.device_info: Optional[dict] = None
        self._sdk_available = False
    
    def _get_config(self) -> OrbbecConfig:
        """Get Orbbec configuration."""
        return get_config().orbbec
    
    def _check_sdk(self) -> bool:
        """Verify pyorbbecsdk is installed."""
        try:
            import pyorbbecsdk
            self._sdk_available = True
            return True
        except ImportError:
            print_error("pyorbbecsdk not found")
            print_info("Install with: pip install pyorbbecsdk2")
            return False
    
    def _check_udev_rules(self) -> bool:
        """Check if udev rules are installed for USB access."""
        udev_paths = [
            "/etc/udev/rules.d/99-obsensor-libusb.rules",
            "/etc/udev/rules.d/99-orbbec.rules"
        ]
        return any(os.path.exists(p) for p in udev_paths)
    
    def verify_connection(self) -> bool:
        """Verify camera USB connection."""
        print_step(1, "Verifying Camera Connection")
        
        if not self._check_sdk():
            return False
        
        cfg = self._get_config()
        result = run_command(["lsusb"], check=False)
        
        if result and cfg.vendor_id in result.lower():
            print_success("Orbbec camera detected on USB")
            
            if not self._check_udev_rules():
                print_warning("udev rules may not be installed")
                print_info("Run: sudo bash ./install_udev_rules.sh from pyorbbecsdk/scripts")
            
            return True
        
        print_error("No Orbbec camera detected on USB")
        return False
    
    def discover(self) -> Optional[str]:
        """Discover connected Orbbec cameras using SDK v2 API."""
        print_step(1, "Discovering Orbbec Cameras")
        
        if not self._check_sdk():
            return None
        
        try:
            from pyorbbecsdk import Context
            
            ctx = Context()
            device_list = ctx.query_devices()
            device_count = device_list.get_count()
            
            if device_count == 0:
                print_warning("No Orbbec devices found via SDK")
                return None
            
            print_success(f"Found {device_count} device(s)")
            
            device = device_list.get_device_by_index(0)
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
    
    # ========================================================================
    # REQUIRED ABSTRACT METHODS
    # ========================================================================
    
    def get_menu_options(self) -> dict:
        """Return menu options for Orbbec camera operations."""
        return {
            "1": "Discover & Get Device Info",
            "2": "View Depth Stream",
            "3": "View Color Stream",
            "4": "View Combined Streams",
            "5": "View 3D Point Cloud (ROS2/RViz2)",
            "6": "Save Point Cloud (.ply)",
            "0": "Back to Main Menu"
        }
    
    def handle_option(self, option: str) -> bool:
        """
        Handle menu option selection.
        Returns True if should return to main menu, False otherwise.
        """
        if option == "1":
            self.discover()
            input("\nPress Enter to continue...")
            return False
            
        elif option == "2":
            self.visualize_depth()
            return False
            
        elif option == "3":
            self.visualize_color()
            return False
            
        elif option == "4":
            self.visualize()
            return False
            
        elif option == "5":
            self.visualize_point_cloud()
            return False
            
        elif option == "6":
            self._save_pointcloud()
            return False
            
        elif option == "0":
            return True
            
        else:
            print_error("Invalid option")
            return False
    
    # ========================================================================
    # VISUALIZATION METHODS
    # ========================================================================
    
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
        """Launch 3D point cloud visualization using ROS2."""
        print_header("Launching Point Cloud Viewer")
        print_info("Using ROS2 + RViz2 for visualization")
        self._run_viewer("pointcloud")
    
    def _save_pointcloud(self) -> None:
        """Save point cloud to PLY file."""
        print_header("Save Point Cloud to PLY")
        
        filename = input("Enter filename (default: pointcloud.ply): ").strip()
        if not filename:
            filename = "pointcloud.ply"
        if not filename.endswith(".ply"):
            filename += ".ply"
        
        print_info(f"Saving to: {filename}")
        
        script = f'''
import numpy as np
from pyorbbecsdk import Config, Pipeline, OBSensorType, PointCloudFilter
import struct

print("[>] Initializing camera...")
config = Config()
pipeline = Pipeline()

try:
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    
    pipeline.start(config)
    print("[✓] Pipeline started")
    
    pc_filter = PointCloudFilter()
    pc_filter.set_camera_param(pipeline.get_camera_param())
    
    print("[>] Capturing point cloud...")
    frames = pipeline.wait_for_frames(1000)
    
    if frames:
        points_frame = pc_filter.process(frames)
        
        if points_frame:
            points = np.frombuffer(points_frame.get_data(), dtype=np.float32).reshape(-1, 3)
            points = points * 0.001  # Convert to meters
            
            # Filter valid points
            valid = (np.abs(points[:, 2]) > 0.01) & (np.abs(points[:, 2]) < 5.0)
            points = points[valid]
            
            print(f"[>] Writing {{len(points)}} points to {filename}")
            
            # Write PLY file
            with open("{filename}", "w") as f:
                f.write("ply\\n")
                f.write("format ascii 1.0\\n")
                f.write(f"element vertex {{len(points)}}\\n")
                f.write("property float x\\n")
                f.write("property float y\\n")
                f.write("property float z\\n")
                f.write("end_header\\n")
                
                for p in points:
                    f.write(f"{{p[0]}} {{p[1]}} {{p[2]}}\\n")
            
            print("[✓] Point cloud saved successfully")
        else:
            print("[!] Failed to generate point cloud")
    else:
        print("[!] Failed to capture frames")
        
finally:
    pipeline.stop()
'''
        
        try:
            exec(script, {{"__name__": "__main__"}})
            input("\nPress Enter to continue...")
        except Exception as e:
            print_error(f"Failed to save point cloud: {{e}}")
            input("\nPress Enter to continue...")
    
    def _run_viewer(self, mode: str) -> None:
        """Internal method to run different visualization modes."""
        if not self._check_sdk():
            return
        
        if mode != "pointcloud":
            print_info("Press 'q' to exit viewer")
        
        viewer_script = self._generate_viewer_script(mode)
        
        try:
            exec(viewer_script, {"__name__": "__main__"})
        except KeyboardInterrupt:
            print_info("\nViewer closed")
        except Exception as e:
            print_error(f"Viewer error: {e}")
    
def _generate_viewer_script(self, mode: str) -> str:
        """Generate viewer script based on mode (Updated with ROS2 Snap fix)."""
        
        scripts = {
            "depth": '''
import cv2
import numpy as np
from pyorbbecsdk import Config, Pipeline, OBSensorType

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
        
        if data.size != width * height:
            continue
            
        data = data.reshape((height, width))
        data_normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        data_colored = cv2.applyColorMap(data_normalized, cv2.COLORMAP_JET)
        
        cv2.imshow("Orbbec Depth", data_colored)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
''',
            "color": '''
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
        fmt = color_frame.get_format()
        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        
        image = None

        if fmt == OBFormat.MJPG:
            try:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except:
                pass
        elif fmt == OBFormat.RGB:
            if data.size == width * height * 3:
                image = data.reshape((height, width, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif fmt == OBFormat.BGR:
            if data.size == width * height * 3:
                image = data.reshape((height, width, 3))

        if image is None:
            continue
        
        cv2.imshow("Orbbec Color", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
''',
            "combined": '''
import cv2
import numpy as np
from pyorbbecsdk import Config, Pipeline, OBSensorType, OBFormat

config = Config()
pipeline = Pipeline()

try:
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
except:
    print("Warning: Could not configure depth stream")

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
        
        depth_frame = frames.get_depth_frame()
        if depth_frame:
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            if data.size == width * height:
                data = data.reshape((height, width))
                data_norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colored = cv2.applyColorMap(data_norm, cv2.COLORMAP_JET)
                cv2.imshow("Orbbec Depth", depth_colored)
        
        color_frame = frames.get_color_frame()
        if color_frame:
            width = color_frame.get_width()
            height = color_frame.get_height()
            fmt = color_frame.get_format()
            data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            
            image = None
            if fmt == OBFormat.MJPG:
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            elif fmt == OBFormat.RGB and data.size == width * height * 3:
                image = data.reshape((height, width, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif fmt == OBFormat.BGR and data.size == width * height * 3:
                image = data.reshape((height, width, 3))
            
            if image is not None:
                cv2.imshow("Orbbec Color", image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
''',
            "pointcloud": '''
#!/usr/bin/env python3
import subprocess
import time
import os
import signal
import sys
import shutil

print("=" * 60)
print("   ROS2 Orbbec Point Cloud Viewer (Fixed)")
print("=" * 60)

# --- ENVIRONMENT FIX FOR SNAP/RVIZ CONFLICTS ---
# This fixes "symbol lookup error: ... libpthread.so.0"
env = os.environ.copy()
if "GTK_PATH" in env:
    print("[>] Removing conflicting GTK_PATH from environment")
    del env["GTK_PATH"]

# Check if ROS2 is accessible
if not shutil.which("ros2"):
    print("[!] 'ros2' command not found.")
    print("[!] Please source your ROS2 installation first (e.g., source /opt/ros/humble/setup.bash)")
    sys.exit(1)

# Check for workspace setup
workspace_setup = os.path.expanduser("~/Desktop/CPSPERRO/my_ros2_ws/install/setup.bash")
if not os.path.exists(workspace_setup):
    print(f"[!] Workspace setup not found at: {workspace_setup}")
    print("[!] Please build your workspace first: colcon build")
    sys.exit(1)

processes = []

def cleanup(signum=None, frame=None):
    print("\\n[>] Shutting down...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=3)
        except:
            p.kill()
    print("[✓] Cleanup complete")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

try:
    print("[>] Launching Orbbec camera node...")
    # We use the current environment (env) instead of hardcoding the source command
    # causing conflicts. We only source the local workspace.
    camera_cmd = f"source {workspace_setup} && ros2 launch orbbec_camera gemini_330_series.launch.py"
    
    camera_process = subprocess.Popen(
        camera_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
        executable="/bin/bash",
        env=env  # Use sanitized environment
    )
    processes.append(camera_process)
    print("[✓] Camera node starting...")
    
    print("[>] Waiting for camera initialization (5s)...")
    time.sleep(5)
    
    # Create temp RViz config
    rviz_config = "/tmp/orbbec_pointcloud.rviz"
    rviz_config_content = """Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 1
      Class: rviz_default_plugins/PointCloud2
      Name: PointCloud2
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /camera/depth/points
      Color: 255; 255; 255
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: true
      Position Transformer: XYZ
      Size (Pixels): 3
      Size (m): 0.01
      Style: Points
    - Class: rviz_default_plugins/Image
      Name: RGB Image
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /camera/color/image_raw
      Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: camera_link
    Frame Rate: 30
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2.5
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.06
        Stereo Focal Distance: 1.0
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Size: 0.05
      Name: Current View
      Yaw: 0.785398
      Pitch: 0.785398
"""
    
    with open(rviz_config, 'w') as f:
        f.write(rviz_config_content)
    
    print("[>] Launching RViz2...")
    rviz_cmd = f"rviz2 -d {rviz_config}"
    
    rviz_process = subprocess.Popen(
        rviz_cmd,
        shell=True,
        preexec_fn=os.setsid,
        executable="/bin/bash",
        env=env # Use sanitized environment
    )
    processes.append(rviz_process)
    print("[✓] RViz2 launched")
    
    print("\\n" + "=" * 60)
    print("[✓] Point Cloud Viewer Running!")
    print("=" * 60)
    print("Press Ctrl+C to exit")
    print("=" * 60)
    
    rviz_process.wait()
    
except KeyboardInterrupt:
    print("\\n[>] Interrupted by user")
except Exception as e:
    print(f"[!] Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cleanup()
'''
        }
        
        return scripts.get(mode, "")
    
    def save_point_cloud(self, filename: str) -> None:
        """Capture and save a single point cloud frame (SDK v2)."""
        print_header("Saving Point Cloud")
        
        if not self._check_sdk():
            return
        
        try:
            import open3d as o3d
        except ImportError:
            print_error("Open3D required: pip install open3d")
            return
        
        config = get_config()
        output_dir = os.path.expanduser(config.general.recording_dir)
        ensure_directory(output_dir)
        output_path = os.path.join(output_dir, f"{filename}.ply")
        
        print_info(f"Saving to: {output_path}")
        
        try:
            from pyorbbecsdk import Config, Pipeline, OBSensorType, PointCloudFilter
            
            cfg = Config()
            pipeline = Pipeline()
            
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            cfg.enable_stream(depth_profiles.get_default_video_stream_profile())
            
            pipeline.start(cfg)
            
            # SDK v2: Use PointCloudFilter
            pc_filter = PointCloudFilter()
            pc_filter.set_camera_param(pipeline.get_camera_param())
            
            # Warm up
            for _ in range(10):
                pipeline.wait_for_frames(100)
            
            frames = pipeline.wait_for_frames(1000)
            if frames:
                points_frame = pc_filter.process(frames)
                
                if points_frame is not None:
                    # Convert to numpy
                    points_data = np.frombuffer(points_frame.get_data(), dtype=np.float32).reshape(-1, 3)
                    points_data = points_data * 0.001 # Convert mm to meters
                    
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_data)
                    
                    o3d.io.write_point_cloud(output_path, pcd)
                    print_success(f"Point cloud saved: {output_path}")
                else:
                    print_error("Failed to generate point cloud data")
            
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
            ("0", "Back to Main Menu"),
        ]
    
    def handle_option(self, option: str) -> None:
        """Handle camera menu option selection."""
        handlers = {
            "1": self.discover,
            "2": self.visualize_depth,
            "3": self.visualize_color,
            "4": self.visualize,
            "5": self.visualize_point_cloud,
        }
        
        if option in handlers:
            handlers[option]()
        elif option == "6":
            filename = input("Enter filename for point cloud: ").strip()
            if filename:
                self.save_point_cloud(filename)


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
        print("  [c] Show Current Configuration")
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
    
    def print_configuration(self) -> None:
        """Display current configuration."""
        print_header("Current Configuration")
        
        config = get_config()
        
        print("\n[Ouster LiDAR]")
        print(f"  Interface:    {config.ouster.ethernet_interface}")
        print(f"  Computer IP:  {config.ouster.computer_ip}/{config.ouster.cidr}")
        print(f"  Sensor IP:    {config.ouster.sensor_ip}")
        print(f"  LiDAR Port:   {config.ouster.lidar_port}")
        print(f"  IMU Port:     {config.ouster.imu_port}")
        
        print("\n[ROS 2]")
        print(f"  Workspace:    {config.ros2.workspace}")
        print(f"  ROS Setup:    {config.ros2.ros_setup}")
        
        print("\n[GLIM SLAM]")
        print(f"  Config Path:  {config.glim.config_path}")
        print(f"  RViz Config:  {config.glim.rviz_config}")
        print(f"  Topics:       {config.glim.pointcloud_topic}, {config.glim.imu_topic}")
        
        print("\n[General]")
        print(f"  Recording Dir: {config.general.recording_dir}")
    
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
            
            choice = input("\nSelect option: ").strip().lower()
            
            if choice == "0":
                print_info("Exiting. Goodbye!")
                break
            elif choice == "c":
                self.print_configuration()
                input("\nPress Enter to continue...")
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
                print_warning("Invalid selection.")
            
            print()


# ==============================================================================
# CLI ARGUMENT PARSING
# ==============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sensor Operations Tool for Ouster LiDAR and Orbbec Camera",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default/config file settings
  %(prog)s --config my_config.yaml            # Use custom config file
  %(prog)s --lidar-ip 169.254.41.35           # Override LiDAR IP
  %(prog)s --ros-ws /home/user/catkin_ws      # Override ROS workspace
  %(prog)s --interface eth0 --computer-ip 169.254.41.100

Configuration Priority:
  1. Command line arguments (highest)
  2. Config file (sensor_ops_config.yaml)
  3. Default values (lowest)
"""
    )
    
    parser.add_argument(
        '--config', '-c',
        default='sensor_ops_config.yaml',
        help='Path to YAML configuration file (default: sensor_ops_config.yaml)'
    )
    
    # Ouster configuration
    ouster_group = parser.add_argument_group('Ouster LiDAR Options')
    ouster_group.add_argument(
        '--lidar-ip',
        help='LiDAR sensor IP address'
    )
    ouster_group.add_argument(
        '--computer-ip',
        help='Computer IP address for LiDAR communication'
    )
    ouster_group.add_argument(
        '--interface', '-i',
        help='Network interface for LiDAR (e.g., eno1, eth0)'
    )
    
    # ROS configuration
    ros_group = parser.add_argument_group('ROS 2 Options')
    ros_group.add_argument(
        '--ros-ws',
        help='ROS 2 workspace path'
    )
    ros_group.add_argument(
        '--ros-setup',
        help='ROS 2 setup.bash path'
    )
    
    return parser.parse_args()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    """Application entry point."""
    global CONFIG
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    print_header("Loading Configuration")
    
    # Try to find config file
    config_paths = [
        args.config,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensor_ops_config.yaml'),
        os.path.expanduser('~/.config/sensor_ops/config.yaml'),
        '/etc/sensor_ops/config.yaml',
    ]
    
    config_loaded = False
    for config_path in config_paths:
        if os.path.exists(config_path):
            CONFIG = AppConfig.from_yaml(config_path)
            config_loaded = True
            break
    
    if not config_loaded:
        print_warning("No config file found. Using defaults.")
        print_info(f"Create config file at: {args.config}")
        CONFIG = AppConfig()
    
    # Apply CLI overrides
    CONFIG.apply_cli_overrides(args)
    
    # Run application
    app = SensorOpsApp()
    
    try:
        app.run()
    except KeyboardInterrupt:
        print_info("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
