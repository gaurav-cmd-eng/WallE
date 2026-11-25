/*
  WALL-E ESP32 Controller Firmware — Physical Sensor Integration & Command Server.

  COMMUNICATION PROTOCOL SPECIFICATION:
  FastAPI and ESP32 exchange newline-delimited JSON messages over Serial (115200 baud) or HTTP.

  Command Format (Host -> ESP32):
    {"cmd": "STOP"|"FORWARD"|"REVERSE"|"LEFT"|"RIGHT"|"SET_SPEED", "speed": 0-255, "seq": int, "ts": float}

  Response Format (ESP32 -> Host):
    {"status": "ok"|"error", "cmd": "<command>", "speed": <speed>, "seq": <seq>, "state": "RUNNING"|"STOPPED"}

  Telemetry Format (ESP32 -> Host, periodic at 10 Hz):
    {"type":"telemetry","distance_cm":45.2,"obstacle":false,"imu":{"ax":0.02,"ay":-0.01,"az":0.98,"gx":0.1,"gy":-0.2,"gz":0.0},"timestamp_ms":12345}

  Safety Guarantees:
  - STOP command immediately forces speed = 0 and halts motors.
  - STOP has absolute priority over every movement command.
  - Watchdog: If no command is received within 1.0 second during motion, controller halts (STOP).
  - Physical motor pins are guarded until motor driver wiring verification.
  - Sensor echo timeout is bounded (25ms) so sensor absence/failure cannot block motor loop.
*/
#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>

const char* WIFI_SSID = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

WebServer server(80);

unsigned long lastCommandTime = 0;
const unsigned long WATCHDOG_TIMEOUT_MS = 1000;
String currentCommand = "STOP";
int currentSpeed = 0;

// ============================================================================
// SENSOR PIN ASSIGNMENTS & HARDWARE CONFIGURATION
// ============================================================================
// HC-SR04 Ultrasonic Sensor pins
const int PIN_TRIG = 5;
const int PIN_ECHO = 18;  // Echo pin MUST connect via 5V->3.3V voltage divider to ESP32!

// MPU6050 6-DoF IMU I2C pins
const int PIN_SDA = 21;
const int PIN_SCL = 22;
const uint8_t MPU6050_ADDR = 0x68;
const uint8_t MPU6050_PWR_MGMT_1 = 0x6B;
const uint8_t MPU6050_ACCEL_XOUT_H = 0x3B;

// Telemetry & Timing Configuration
const unsigned long ECHO_TIMEOUT_US = 25000;       // 25ms maximum echo timeout (~4.3m)
const float OBSTACLE_DISTANCE_THRESHOLD_CM = 30.0;  // Proximity threshold for obstacle flag
const unsigned long TELEMETRY_INTERVAL_MS = 100;    // 10 Hz telemetry rate (every 100ms)
unsigned long lastTelemetryTime = 0;

bool mpuInitialized = false;

// ============================================================================
// MPU6050 I2C DRIVER (NON-BLOCKING / GRACEFUL FAILOVER)
// ============================================================================
struct IMUData {
  bool valid;
  float ax;
  float ay;
  float az;
  float gx;
  float gy;
  float gz;
};

void setupMPU6050() {
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.beginTransmission(MPU6050_ADDR);
  byte error = Wire.endTransmission();
  if (error == 0) {
    // Wake up MPU6050 from sleep mode by writing 0 to PWR_MGMT_1 register
    Wire.beginTransmission(MPU6050_ADDR);
    Wire.write(MPU6050_PWR_MGMT_1);
    Wire.write(0x00);
    byte pwrErr = Wire.endTransmission();
    if (pwrErr == 0) {
      mpuInitialized = true;
    } else {
      mpuInitialized = false;
    }
  } else {
    mpuInitialized = false;
  }
}

IMUData readMPU6050() {
  IMUData imu = {false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  if (!mpuInitialized) {
    return imu;
  }

  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) {
    return imu;
  }

  // Request 14 bytes: Accel X/Y/Z (6), Temp (2), Gyro X/Y/Z (6)
  uint8_t bytesRead = Wire.requestFrom(MPU6050_ADDR, (uint8_t)14, (uint8_t)true);
  if (bytesRead == 14) {
    int16_t rawAx = (Wire.read() << 8) | Wire.read();
    int16_t rawAy = (Wire.read() << 8) | Wire.read();
    int16_t rawAz = (Wire.read() << 8) | Wire.read();
    int16_t rawTemp = (Wire.read() << 8) | Wire.read(); // temperature (available if needed)
    int16_t rawGx = (Wire.read() << 8) | Wire.read();
    int16_t rawGy = (Wire.read() << 8) | Wire.read();
    int16_t rawGz = (Wire.read() << 8) | Wire.read();

    // Convert raw ADC readings to physical units:
    // Accel sensitivity (+/-2g scale): 16384 LSB/g
    // Gyro sensitivity (+/-250 deg/s scale): 131 LSB/(deg/s)
    imu.valid = true;
    imu.ax = rawAx / 16384.0f;
    imu.ay = rawAy / 16384.0f;
    imu.az = rawAz / 16384.0f;
    imu.gx = rawGx / 131.0f;
    imu.gy = rawGy / 131.0f;
    imu.gz = rawGz / 131.0f;
  }
  return imu;
}

// ============================================================================
// HC-SR04 ULTRASONIC RANGING (BOUNDED TIMEOUT)
// ============================================================================
struct UltrasonicData {
  bool valid;
  float distance_cm;
  bool obstacle;
};

UltrasonicData readUltrasonic() {
  UltrasonicData u = {false, -1.0f, false};

  // Generate clean 10-microsecond trigger pulse
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  // Bounded echo timeout: 25000 us (~4.3m). Never hangs loop if sensor disconnected.
  unsigned long duration = pulseIn(PIN_ECHO, HIGH, ECHO_TIMEOUT_US);

  if (duration > 0) {
    // Speed of sound: 343 m/s = 0.0343 cm/us. Divide by 2 for round-trip distance.
    float dist = (duration * 0.0343f) / 2.0f;
    if (dist >= 2.0f && dist <= 400.0f) {
      u.valid = true;
      u.distance_cm = dist;
      u.obstacle = (dist <= OBSTACLE_DISTANCE_THRESHOLD_CM);
    }
  }
  return u;
}

// ============================================================================
// SENSOR TELEMETRY BROADCAST (10 Hz STREAM)
// ============================================================================
void broadcastTelemetry() {
  unsigned long now = millis();
  if (now - lastTelemetryTime < TELEMETRY_INTERVAL_MS) {
    return;
  }
  lastTelemetryTime = now;

  UltrasonicData us = readUltrasonic();
  IMUData imu = readMPU6050();

  // Format newline-delimited JSON telemetry packet
  String json = "{\"type\":\"telemetry\",";

  if (us.valid) {
    json += "\"distance_cm\":" + String(us.distance_cm, 1) + ",";
    json += "\"obstacle\":" + String(us.obstacle ? "true" : "false") + ",";
  } else {
    json += "\"distance_cm\":null,";
    json += "\"obstacle\":false,";
  }

  if (imu.valid) {
    json += "\"imu\":{";
    json += "\"ax\":" + String(imu.ax, 3) + ",";
    json += "\"ay\":" + String(imu.ay, 3) + ",";
    json += "\"az\":" + String(imu.az, 3) + ",";
    json += "\"gx\":" + String(imu.gx, 2) + ",";
    json += "\"gy\":" + String(imu.gy, 2) + ",";
    json += "\"gz\":" + String(imu.gz, 2);
    json += "},";
  } else {
    json += "\"imu\":null,";
  }

  json += "\"timestamp_ms\":" + String(now);
  json += "}";

  Serial.println(json);
}

// ============================================================================
// DUAL-CHANNEL MOTOR DRIVER ABSTRACTION
// ============================================================================
// Supports standard 2-channel H-Bridge architectures (L298N, TB6612FNG, DRV8833, L9110S).
// Strictly guarded: pins default to -1 (UNASSIGNED) and enabled defaults to false.
// No physical pins are driven until the driver model is purchased and verified.

enum MotorDirection {
  MOTOR_DIR_STOP = 0,
  MOTOR_DIR_FORWARD,
  MOTOR_DIR_REVERSE
};

struct MotorChannelOutput {
  MotorDirection dirLeft;
  int speedLeft;
  MotorDirection dirRight;
  int speedRight;
};

struct MotorDriverConfig {
  // Pin definitions (-1 indicates unassigned)
  int pinLeftPWM = -1;      // ENA on L298N or PWMA on TB6612FNG
  int pinLeftDir1 = -1;     // IN1
  int pinLeftDir2 = -1;     // IN2
  int pinRightPWM = -1;     // ENB on L298N or PWMB on TB6612FNG
  int pinRightDir1 = -1;    // IN3
  int pinRightDir2 = -1;    // IN4
  int pinStandby = -1;      // STBY pin (TB6612FNG requires HIGH, -1 if unused)

  // Safety latch: must be explicitly set to true after wiring is verified
  bool enabled = false;
};

MotorDriverConfig motorConfig;

MotorChannelOutput computeDifferentialDrive(String cmd, int speed) {
  MotorChannelOutput out = {MOTOR_DIR_STOP, 0, MOTOR_DIR_STOP, 0};
  int spd = constrain(speed, 0, 255);

  if (cmd == "FORWARD") {
    out.dirLeft = MOTOR_DIR_FORWARD;
    out.speedLeft = spd;
    out.dirRight = MOTOR_DIR_FORWARD;
    out.speedRight = spd;
  } else if (cmd == "REVERSE" || cmd == "BACKWARD") {
    out.dirLeft = MOTOR_DIR_REVERSE;
    out.speedLeft = spd;
    out.dirRight = MOTOR_DIR_REVERSE;
    out.speedRight = spd;
  } else if (cmd == "LEFT") {
    // Skid-steer differential rotation: Left motor reverse, Right motor forward
    out.dirLeft = MOTOR_DIR_REVERSE;
    out.speedLeft = spd;
    out.dirRight = MOTOR_DIR_FORWARD;
    out.speedRight = spd;
  } else if (cmd == "RIGHT") {
    // Skid-steer differential rotation: Left motor forward, Right motor reverse
    out.dirLeft = MOTOR_DIR_FORWARD;
    out.speedLeft = spd;
    out.dirRight = MOTOR_DIR_REVERSE;
    out.speedRight = spd;
  } else {
    // STOP or unrecognized command
    out.dirLeft = MOTOR_DIR_STOP;
    out.speedLeft = 0;
    out.dirRight = MOTOR_DIR_STOP;
    out.speedRight = 0;
  }
  return out;
}

void applyMotorOutputs(const MotorChannelOutput& out) {
  // STRICT SAFETY ENFORCEMENT: Never write to pins if driver is unconfigured
  if (!motorConfig.enabled) {
    return;
  }

  // Once physical driver arrives and motorConfig.enabled is set to true:
  // Hardware PWM / GPIO driver logic will attach to pinLeftPWM, pinLeftDir1, etc.
}

// ============================================================================
// MOTOR COMMAND EXECUTION & SERIAL PROTOCOL
// ============================================================================
void executeMotorCommand(String cmd, int speed) {
  cmd.toUpperCase();
  if (cmd == "FORWARD") {
    UltrasonicData us = readUltrasonic();
    if (us.valid && us.obstacle) {
      currentCommand = "STOP";
      currentSpeed = 0;
      lastCommandTime = millis();
      MotorChannelOutput halt = computeDifferentialDrive("STOP", 0);
      applyMotorOutputs(halt);
      return;
    }
  }

  if (cmd == "STOP" || speed <= 0) {
    currentCommand = "STOP";
    currentSpeed = 0;
  } else {
    currentCommand = cmd;
    currentSpeed = constrain(speed, 0, 255);
  }
  lastCommandTime = millis();

  MotorChannelOutput out = computeDifferentialDrive(currentCommand, currentSpeed);
  applyMotorOutputs(out);
}

void handleDrive() {
  String direction = server.arg("direction");
  String speedStr = server.arg("speed");
  int speed = speedStr.length() > 0 ? speedStr.toInt() : 150;
  executeMotorCommand(direction, speed);
  Serial.printf("{\"status\":\"ok\",\"cmd\":\"%s\",\"speed\":%d,\"transport\":\"http\"}\n", currentCommand.c_str(), currentSpeed);
  server.send(200, "application/json", "{\"ok\":true,\"status\":\"ok\",\"note\":\"drive command received\"}");
}

void processSerialCommand() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() == 0) return;

    // Simple JSON field extraction (supports {"cmd":"...","speed":...,"seq":...})
    String cmd = "STOP";
    int speed = 0;
    int seq = 0;

    int cmdIdx = line.indexOf("\"cmd\":");
    if (cmdIdx >= 0) {
      int startQuote = line.indexOf("\"", cmdIdx + 6);
      int endQuote = line.indexOf("\"", startQuote + 1);
      if (startQuote >= 0 && endQuote > startQuote) {
        cmd = line.substring(startQuote + 1, endQuote);
      }
    }

    int speedIdx = line.indexOf("\"speed\":");
    if (speedIdx >= 0) {
      int comma = line.indexOf(",", speedIdx);
      int brace = line.indexOf("}", speedIdx);
      int endIdx = (comma >= 0 && comma < brace) ? comma : brace;
      if (endIdx > speedIdx) {
        speed = line.substring(speedIdx + 8, endIdx).toInt();
      }
    }

    int seqIdx = line.indexOf("\"seq\":");
    if (seqIdx >= 0) {
      int comma = line.indexOf(",", seqIdx);
      int brace = line.indexOf("}", seqIdx);
      int endIdx = (comma >= 0 && comma < brace) ? comma : brace;
      if (endIdx > seqIdx) {
        seq = line.substring(seqIdx + 6, endIdx).toInt();
      }
    }

    executeMotorCommand(cmd, speed);

    // Send formatted JSON acknowledgement
    Serial.printf("{\"status\":\"ok\",\"cmd\":\"%s\",\"speed\":%d,\"seq\":%d,\"state\":\"%s\"}\n",
                  currentCommand.c_str(), currentSpeed, seq, (currentCommand == "STOP" ? "STOPPED" : "RUNNING"));
  }
}

void setup() {
  Serial.begin(115200);

  // Initialize HC-SR04 ultrasonic pins
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  digitalWrite(PIN_TRIG, LOW);

  // Initialize MPU6050 I2C bus
  setupMPU6050();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(200);
    Serial.print(".");
    attempts++;
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.localIP());
    server.on("/drive", HTTP_GET, handleDrive);
    server.begin();
  } else {
    Serial.println("Wi-Fi not connected (Serial control ready)");
  }
  Serial.println("WALL-E command server ready.");
  lastCommandTime = millis();
  lastTelemetryTime = millis();
}

void loop() {
  server.handleClient();
  processSerialCommand();
  broadcastTelemetry();

  // Active motion safety: Halt forward motion immediately if obstacle detected
  if (currentCommand == "FORWARD") {
    UltrasonicData us = readUltrasonic();
    if (us.valid && us.obstacle) {
      executeMotorCommand("STOP", 0);
      Serial.println("{\"status\":\"obstacle\",\"cmd\":\"STOP\",\"state\":\"STOPPED\",\"reason\":\"obstacle_detected\"}");
    }
  }

  // Hardware watchdog: Stop motors if no command received within timeout during motion
  if (currentCommand != "STOP" && (millis() - lastCommandTime > WATCHDOG_TIMEOUT_MS)) {
    executeMotorCommand("STOP", 0);
    Serial.println("{\"status\":\"watchdog\",\"cmd\":\"STOP\",\"state\":\"STOPPED\",\"reason\":\"timeout\"}");
  }
}
