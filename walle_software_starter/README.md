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

### Prerequisites
1. Raspberry Pi 4 with clean Raspberry Pi OS 64-bit (Debian 12 Bookworm).
2. Connect USB Webcam to USB 3.0 port (blue port).
3. Connect ESP32 to USB 2.0 port via micro-USB data cable.

### Quick Setup Script
Run the automated deployment script from the project root:
```bash
chmod +x setup_rpi.sh
./setup_rpi.sh
```

### Manual Installation Steps
```bash
# 1. Install system runtime dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip libgl1 libglib2.0-0 libgomp1 v4l-utils udev

# 2. Add permissions for serial & camera
sudo usermod -aG dialout,video $USER

# 3. Create virtual environment and install packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 4. Verify connected hardware
v4l2-ctl --list-devices     # Look for /dev/video0
ls -l /dev/ttyUSB*          # Look for /dev/ttyUSB0
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

To launch WALL-E automatically when the Raspberry Pi boots:

```bash
# Copy and enable the systemd service
sudo cp walle.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable walle.service
sudo systemctl start walle.service

# Check status and live logs
sudo systemctl status walle.service
sudo journalctl -u walle.service -f
```

---

## 4. ESP32 Firmware Compilation & Flashing Instructions

The ESP32 firmware source is located in `esp32/walle_controller.ino`.

### Option A: Using Arduino IDE
1. Open Arduino IDE (v2.x).
2. Go to **File -> Preferences**. In *Additional boards manager URLs*, add:
   `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
3. In **Tools -> Board -> Boards Manager**, search for `esp32` by Espressif Systems and install it.
4. Select board: **Tools -> Board -> ESP32 Arduino -> ESP32 Dev Module**.
5. Connect ESP32 via USB and select the serial port (**Tools -> Port**).
6. Open `walle_controller.ino` and click **Upload** (Ctrl + U).

### Option B: Using Arduino CLI (Terminal)
```bash
# 1. Download and configure arduino-cli
arduino-cli config init
arduino-cli config add board_manager.additional_urls https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
arduino-cli core update-index
arduino-cli core install esp32:esp32

# 2. Compile sketch
arduino-cli compile --fqbn esp32:esp32:esp32 walle_software_starter/esp32/walle_controller.ino

# 3. Upload to ESP32
arduino-cli upload -p /dev/ttyUSB0 --fqbn esp32:esp32:esp32 walle_software_starter/esp32/walle_controller.ino
```

---

## 5. Dual-Channel Motor Driver Interface

The firmware provides a configurable differential drive interface ready for dual-channel H-Bridge drivers:
- **Left Channel**: `pinLeftPWM`, `pinLeftDir1`, `pinLeftDir2`
- **Right Channel**: `pinRightPWM`, `pinRightDir1`, `pinRightDir2`
- **Standby**: `pinStandby` (required HIGH for TB6612FNG)
- **Safety Lock**: `motorConfig.enabled` defaults to `false` and pins default to `-1` until physical driver verification.

When the driver is mounted, assign the pins in `walle_controller.ino`:
```cpp
motorConfig.pinLeftPWM  = 14;
motorConfig.pinLeftDir1 = 27;
motorConfig.pinLeftDir2 = 26;
motorConfig.pinRightPWM = 32;
motorConfig.pinRightDir1= 25;
motorConfig.pinRightDir2= 33;
motorConfig.pinStandby  = 13;
motorConfig.enabled     = true;
```

---

## 6. Safety Architecture Summary

1. **Obstacle Stop Priority**: Live HC-SR04 ultrasonic distance $\le 30.0\text{ cm}$ immediately commands STOP.
2. **Persistent Emergency Latch**: Manual STOP locks safety latch until explicit `/api/safety/reset`.
3. **Hardware Watchdog**: ESP32 halts motors if serial communication ceases for $> 1.0\text{ s}$.
4. **Stale Telemetry Motion Guard**: Forward motion is terminated if telemetry pauses for $> 1.0\text{ s}$.
5. **No Sudden Lurch**: Obstacle recovery commands `HOLD` before allowing forward approach.

---

## 7. Stage 6 & 6B: 2D Simulation Arena & Autonomous Detours

The dashboard includes a dedicated real-time **2D Simulation Arena** designed for demonstrations and academic evaluation:
- **Interactive Following**: WASD / Arrow Keys / Canvas-click movement with target reticle.
- **Following Distance Maintenance**: Holds distance at ~1.5m within the green zone (1.2m - 1.8m).
- **Stage 6B Autonomous Detours**: Plans collision-free waypoint detours around obstacles using 2D geometric clearance.
- **Stage 6B 360-Degree Target Search**: In-place rotation scan ($45^\circ/\text{s}$, $0.0\,\text{m/s}$ linear) that prevents target hijacking and reacquires the locked person when they step outside camera view.
- **Synthetic Video Feed Standby**: Generates annotated HUD streaming over `/video_feed` when no physical webcam is plugged in.

---

## 8. Verification & Test Suite

Run the full regression test suite:
```bash
python -m unittest discover -s . -p "test_*.py"
```
**Results: 190 / 190 tests passing (100% pass rate in ~2.5s, 0 failures, 0 errors).**
