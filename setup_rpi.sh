#!/usr/bin/env bash
# ============================================================================
# WALL-E Robot Rover — Raspberry Pi 4 Setup & Deployment Script
# Target OS: Raspberry Pi OS 64-bit (Debian 12 Bookworm)
# ============================================================================

set -e

echo "=== [1/6] Updating APT Package Lists & Installing System Dependencies ==="
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    v4l-utils \
    udev

echo "=== [2/6] Configuring Hardware Group Permissions (Dialout & Video) ==="
# Add current user to dialout (for ESP32 USB-serial) and video (for USB webcam)
sudo usermod -aG dialout "$USER"
sudo usermod -aG video "$USER"

echo "=== [3/6] Configuring udev Rules for ESP32 and USB Camera ==="
# Standard CP210x, CH340, and FTDI USB-UART converters for ESP32
sudo tee /etc/udev/rules.d/99-walle-hardware.rules > /dev/null << 'EOF'
# ESP32 USB Serial Converters (CP210x, CH340, FTDI)
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE="0666", GROUP="dialout"
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", MODE="0666", GROUP="dialout"

# USB Video Cameras
SUBSYSTEM=="video4linux", MODE="0666", GROUP="video"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

echo "=== [4/6] Creating Python Virtual Environment & Installing Wheels ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "=== [5/6] Verifying YOLOv8n ONNX Model & Files ==="
if [ ! -f "models/yolov8n.onnx" ] && [ -f "walle_software_starter/models/yolov8n.onnx" ]; then
    mkdir -p models
    cp walle_software_starter/models/yolov8n.onnx models/yolov8n.onnx
fi

if [ -f "models/yolov8n.onnx" ]; then
    echo "Model verified: models/yolov8n.onnx ($(du -h models/yolov8n.onnx | cut -f1))"
else
    echo "WARNING: models/yolov8n.onnx missing. Please ensure model file is placed in models/."
fi

echo "=== [6/6] Verifying Hardware Devices ==="
echo "Connected Video Devices:"
v4l2-ctl --list-devices 2>/dev/null || ls -l /dev/video* 2>/dev/null || echo "No video devices found yet."

echo "Connected Serial Devices:"
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "No ESP32 USB serial device found yet."

echo ""
echo "============================================================================"
echo "WALL-E Setup Complete!"
echo "To start the application manually:"
echo "    source .venv/bin/activate"
echo "    export ESP32_PORT=/dev/ttyUSB0"
echo "    export CAMERA_INDEX=0"
echo "    uvicorn main:app --host 0.0.0.0 --port 8000"
echo ""
echo "To install as a systemd service running on boot:"
echo "    sudo cp walle.service /etc/systemd/system/"
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl enable walle.service"
echo "    sudo systemctl start walle.service"
echo "============================================================================"
