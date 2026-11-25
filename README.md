# WALL-E Autonomous Rover — Control Center & Integration Guide

A unified FastAPI backend, YOLOv8n vision tracking pipeline, dual-mode follower, and ESP32 hardware telemetry/command hub for the WALL-E robotics prototype.

---

## 1. System Architecture & Hardware Connections

### Hardware Overview
- **Compute (Vision & Brain)**: Raspberry Pi 4 (4GB RAM) running Raspberry Pi OS 64-bit (Debian Bookworm).
- **Embedded Controller (Real-time I/O)**: ESP32 DevKit V1 connected to Raspberry Pi via USB Serial (`/dev/ttyUSB0` at 115200 baud).
- **Vision**: USB Webcam running OpenCV DNN with YOLOv8n ONNX (`models/yolov8n.onnx`).
- **Obstacle Proximity Sensing**: HC-SR04 Ultrasonic Sensor connected to ESP32:
  - `VCC`: 5V (from 5V rail / power supply)
  - `GND`: Common Ground
  - `TRIG`: ESP32 GPIO 5 (3.3V logic compatible)
  - `ECHO`: ESP32 GPIO 18 (**CRITICAL**: MUST pass through a 5V -> 3.3V voltage divider: $1\text{k}\Omega$ series, $2\text{k}\Omega$ to GND, or level shifter).
- **Inertial Measurement (IMU)**: MPU6050 6-DoF accelerometer & gyroscope connected to ESP32:
  - `VCC`: 3.3V or 5V (module has onboard regulator)
  - `GND`: Common Ground
  - `SDA`: ESP32 GPIO 21 (I2C Data)
  - `SCL`: ESP32 GPIO 22 (I2C Clock)
- **Drive Actuators**: Two yellow TT geared DC motors (skid-steer differential drive).
  - *Motor driver pending purchase*: TB6612FNG or L298N.
  - Driver abstraction is implemented in firmware; physical motor pins remain unassigned until the driver is mounted.
- **Power**: boAt 20,000mAh Power Bank (5V / 3A output) supplying Raspberry Pi 4 and ESP32 via USB-C.

---

## 2. Raspberry Pi 4 Deployment Instructions

### Quick Setup Script
```bash
chmod +x setup_rpi.sh
./setup_rpi.sh
```

### Starting the Server
```bash
source .venv/bin/activate
export ESP32_PORT=/dev/ttyUSB0
export CAMERA_INDEX=0
uvicorn main:app --host 0.0.0.0 --port 8000
```
Open `http://<RASPBERRY_PI_IP>:8000` from any browser on the local network.

---

## 3. Automated Boot Service (systemd)
```bash
sudo cp walle.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable walle.service
sudo systemctl start walle.service
```

---

## 4. Dual-Channel Motor Driver Interface
The firmware provides a configurable differential drive interface ready for dual-channel H-Bridge drivers (TB6612FNG, L298N, DRV8833). Physical pins default to `-1` and `enabled` defaults to `false` until physical driver verification.

---

## 5. Stage 6: Interactive 2D Simulation Arena (Professor Demonstration)

The dashboard includes a dedicated real-time **2D Simulation Arena** designed for demonstrations and academic evaluation.

### How to Run the Demonstration:
1. Start the server:
   ```bash
   uvicorn walle_software_starter.main:app --host 127.0.0.1 --port 8000
   ```
2. Navigate to `http://127.0.0.1:8000` and locate the **Interactive WALL-E Simulation Arena** panel.
3. Control the simulated person using:
   - Keyboard: <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd> or <kbd>Arrow Keys</kbd>
   - Movement Pad buttons on the dashboard
   - Canvas direct-click
4. Observe WALL-E's autonomous kinematics:
   - Steering alignment (Left / Right)
   - Following distance maintenance (~1.5m target in the green zone)
   - Obstacle override stopping (injected obstacle within 30cm)
   - Target locking & anti-switching when adding/removing people
   - Emergency STOP latching and manual reset to IDLE

### Running Automated Verification Tests:
```bash
python -m unittest -v test_sim_arena.py
python -m unittest discover -s . -p "test_*.py"
```
Full regression suite: **161 / 161 tests passing**.
See [stage_6_demonstration_guide.md](stage_6_demonstration_guide.md) for step-by-step walkthrough scripts.
