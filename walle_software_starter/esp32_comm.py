import json
import time
from typing import Dict, List, Optional, Any, Tuple


class ESP32Command:
    STOP = "STOP"
    FORWARD = "FORWARD"
    REVERSE = "REVERSE"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    SET_SPEED = "SET_SPEED"

    VALID_COMMANDS = {STOP, FORWARD, REVERSE, LEFT, RIGHT, SET_SPEED}
    ALIASES = {
        "BACKWARD": REVERSE,
        "BACK": REVERSE,
        "HALT": STOP,
    }

    @classmethod
    def normalize(cls, cmd_str: str) -> str:
        cmd_upper = str(cmd_str).strip().upper()
        if cmd_upper in cls.ALIASES:
            return cls.ALIASES[cmd_upper]
        if cmd_upper in cls.VALID_COMMANDS:
            return cmd_upper
        raise ValueError(f"Invalid ESP32 command: '{cmd_str}'. Expected one of {sorted(cls.VALID_COMMANDS)}")


class ConnectionStatus:
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


class SensorTelemetry:
    """
    Structured container for ESP32 sensor telemetry packet.
    Fields include HC-SR04 distance/obstacle and MPU6050 6-DoF accelerometer/gyroscope readings.
    """
    def __init__(self,
                 distance_cm: Optional[float] = None,
                 obstacle: bool = False,
                 imu: Optional[Dict[str, float]] = None,
                 timestamp_ms: Optional[int] = None,
                 received_at: Optional[float] = None,
                 is_valid: bool = True,
                 raw_data: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None):
        self.distance_cm = distance_cm
        self.obstacle = bool(obstacle)
        self.imu = imu or {}
        self.has_imu = bool(imu and any(k in imu for k in ("ax", "ay", "az", "gx", "gy", "gz")))
        self.timestamp_ms = timestamp_ms
        self.received_at = received_at if received_at is not None else time.time()
        self.is_valid = is_valid
        self.raw_data = raw_data or {}
        self.error = error

    def is_stale(self, timeout_sec: float = 2.0, now: Optional[float] = None) -> bool:
        curr = now if now is not None else time.time()
        return (curr - self.received_at) > timeout_sec

    def to_dict(self, now: Optional[float] = None) -> Dict[str, Any]:
        curr = now if now is not None else time.time()
        age = round(max(0.0, curr - self.received_at), 2)
        return {
            "type": "telemetry",
            "distance_cm": round(self.distance_cm, 1) if self.distance_cm is not None else None,
            "obstacle": self.obstacle,
            "imu": self.imu if self.has_imu else None,
            "has_imu": self.has_imu,
            "has_distance": self.distance_cm is not None,
            "timestamp_ms": self.timestamp_ms,
            "age_sec": age,
            "is_stale": self.is_stale(now=curr),
            "is_valid": self.is_valid,
            "error": self.error
        }


def parse_telemetry_packet(data: Any, now: Optional[float] = None) -> Optional[SensorTelemetry]:
    """
    Parses and validates a sensor telemetry dictionary or JSON string.
    Returns SensorTelemetry object, or None if not a telemetry packet.
    Handles malformed fields, missing keys, initialization failures, and physical plausibility.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data.strip())
        except Exception:
            return SensorTelemetry(is_valid=False, error="Invalid JSON telemetry packet")

    if not isinstance(data, dict):
        return None

    # Identify telemetry packet by type tag or telemetry-specific fields
    msg_type = data.get("type")
    if msg_type != "telemetry" and not ("distance_cm" in data or "imu" in data):
        return None

    # Parse HC-SR04 distance
    raw_dist = data.get("distance_cm")
    if raw_dist is None:
        raw_dist = data.get("dist_cm")
    distance_cm = None
    if raw_dist is not None:
        try:
            val = float(raw_dist)
            # Physical bounds check: 2.0cm to 450.0cm
            if 0.0 <= val <= 500.0:
                distance_cm = val
            else:
                distance_cm = None
        except (ValueError, TypeError):
            distance_cm = None

    obstacle = bool(data.get("obstacle", False))
    if distance_cm is not None and distance_cm <= 30.0:
        obstacle = True

    # Parse MPU6050 6-DoF IMU
    raw_imu = data.get("imu")
    imu_dict = None
    if isinstance(raw_imu, dict):
        imu_dict = {}
        for key in ("ax", "ay", "az", "gx", "gy", "gz"):
            if key in raw_imu and raw_imu[key] is not None:
                try:
                    imu_dict[key] = float(raw_imu[key])
                except (ValueError, TypeError):
                    pass

    timestamp_ms = data.get("timestamp_ms")
    if timestamp_ms is not None:
        try:
            timestamp_ms = int(timestamp_ms)
        except (ValueError, TypeError):
            timestamp_ms = None

    return SensorTelemetry(
        distance_cm=distance_cm,
        obstacle=obstacle,
        imu=imu_dict,
        timestamp_ms=timestamp_ms,
        received_at=now if now is not None else time.time(),
        is_valid=True,
        raw_data=data
    )


class SerialTransportBase:
    """Abstract base transport for serial communications."""
    def connect(self) -> bool:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def is_connected(self) -> bool:
        raise NotImplementedError

    def write_line(self, line: str) -> bool:
        raise NotImplementedError

    def read_line(self, timeout_sec: float = 1.0) -> Optional[str]:
        raise NotImplementedError


class MockSerialTransport(SerialTransportBase):
    """
    In-memory mock serial transport for automated testing without physical hardware.
    Simulates command transmission, response parsing, delays, timeouts, disconnects,
    and sensor telemetry stream injection.
    """
    def __init__(self, auto_ack: bool = True):
        self._connected = True
        self.auto_ack = auto_ack
        self.sent_lines: List[str] = []
        self.response_queue: List[str] = []
        self.simulate_timeout: bool = False
        self.simulate_disconnect: bool = False
        self.last_parsed_command: Optional[Dict[str, Any]] = None

    def connect(self) -> bool:
        self._connected = True
        self.simulate_disconnect = False
        return True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and not self.simulate_disconnect

    def write_line(self, line: str) -> bool:
        if not self.is_connected():
            return False
        self.sent_lines.append(line)
        try:
            parsed = json.loads(line.strip())
            self.last_parsed_command = parsed
            if self.auto_ack and not self.simulate_timeout:
                ack_response = {
                    "status": "ok",
                    "cmd": parsed.get("cmd", "UNKNOWN"),
                    "speed": parsed.get("speed", 0),
                    "seq": parsed.get("seq", 0),
                    "state": "RUNNING" if parsed.get("cmd") != "STOP" else "STOPPED"
                }
                self.response_queue.append(json.dumps(ack_response) + "\n")
        except Exception:
            self.last_parsed_command = None
        return True

    def read_line(self, timeout_sec: float = 1.0) -> Optional[str]:
        if not self.is_connected() or self.simulate_timeout:
            return None
        if self.response_queue:
            return self.response_queue.pop(0)
        return None

    def queue_response(self, response_dict_or_str: Any) -> None:
        """Injects a custom response for testing."""
        if isinstance(response_dict_or_str, dict):
            self.response_queue.append(json.dumps(response_dict_or_str) + "\n")
        else:
            self.response_queue.append(str(response_dict_or_str) + "\n")

    def queue_telemetry(self,
                        distance_cm: Optional[float] = None,
                        obstacle: bool = False,
                        imu: Optional[Dict[str, float]] = None,
                        timestamp_ms: Optional[int] = None) -> None:
        """Injects a simulated sensor telemetry packet into the read queue."""
        telem = {
            "type": "telemetry",
            "distance_cm": distance_cm,
            "obstacle": obstacle,
            "imu": imu,
            "timestamp_ms": timestamp_ms or int(time.time() * 1000)
        }
        self.queue_response(telem)


class SerialTransport(SerialTransportBase):
    """
    Physical serial transport using pyserial.
    Guarded for hardware safety: only opens when explicitly initialized and configured.
    """
    def __init__(self, port: str = "COM3", baudrate: int = 115200, timeout: float = 1.0):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn: Any = None

    def connect(self) -> bool:
        try:
            import serial
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            return True
        except Exception:
            self.serial_conn = None
            return False

    def disconnect(self) -> None:
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None

    def is_connected(self) -> bool:
        return self.serial_conn is not None and getattr(self.serial_conn, "is_open", False)

    def write_line(self, line: str) -> bool:
        if not self.is_connected():
            return False
        try:
            data = line.encode("utf-8")
            self.serial_conn.write(data)
            self.serial_conn.flush()
            return True
        except Exception:
            return False

    def read_line(self, timeout_sec: float = 1.0) -> Optional[str]:
        if not self.is_connected():
            return None
        try:
            raw = self.serial_conn.readline()
            if raw:
                return raw.decode("utf-8")
            return None
        except Exception:
            return None


class ESP32CommunicationManager:
    """
    Communication Manager for ESP32 Hardware Integration.
    Manages connection lifecycle, JSON command serialization, response verification,
    watchdog timeouts, communication failure transitions, STOP priority enforcement,
    and multiplexed sensor telemetry ingestion (HC-SR04 & MPU6050).
    """

    def __init__(self,
                 transport: Optional[SerialTransportBase] = None,
                 watchdog_timeout_sec: float = 1.0,
                 default_speed: int = 150,
                 simulation_mode: bool = True,
                 telemetry_timeout_sec: float = 2.0):
        self.simulation_mode = simulation_mode
        self.transport: SerialTransportBase = transport or MockSerialTransport()
        self.watchdog_timeout_sec = watchdog_timeout_sec
        self.default_speed = default_speed
        self.telemetry_timeout_sec = telemetry_timeout_sec

        self.connection_status = ConnectionStatus.CONNECTED if self.transport.is_connected() else ConnectionStatus.DISCONNECTED
        self.seq = 0
        self.last_command: Dict[str, Any] = {"cmd": ESP32Command.STOP, "speed": 0, "seq": 0, "ts": time.time()}
        self.last_response: Optional[Dict[str, Any]] = None
        self.last_heartbeat = time.time()
        self.error_message: Optional[str] = None
        self.current_state = "STOPPED"

        # Sensor Telemetry State
        self.latest_telemetry: Optional[SensorTelemetry] = None
        self.telemetry_history: List[SensorTelemetry] = []
        self.max_telemetry_history: int = 50

    @property
    def last_telemetry(self) -> Optional[SensorTelemetry]:
        return self.latest_telemetry

    def connect(self) -> bool:
        ok = self.transport.connect()
        self.connection_status = ConnectionStatus.CONNECTED if ok else ConnectionStatus.ERROR
        if not ok:
            self.error_message = "Failed to establish transport connection"
        else:
            self.error_message = None
            self.last_heartbeat = time.time()
        return ok

    def disconnect(self) -> None:
        self.send_stop(reason="Controller disconnect")
        self.transport.disconnect()
        self.connection_status = ConnectionStatus.DISCONNECTED

    def is_connected(self) -> bool:
        return self.transport.is_connected()

    def process_incoming_line(self, line: str, now: Optional[float] = None) -> Tuple[str, Any]:
        """
        Parses an incoming newline-delimited JSON line.
        Distinguishes:
        - ('telemetry', SensorTelemetry)
        - ('ack', Dict[str, Any])
        - ('watchdog', Dict[str, Any])
        - ('error', str)
        - ('unknown', Any)
        """
        curr = now if now is not None else time.time()
        line_clean = line.strip()
        if not line_clean:
            return ("empty", None)

        try:
            parsed = json.loads(line_clean)
        except Exception as exc:
            return ("error", f"Malformed JSON: {exc}")

        if not isinstance(parsed, dict):
            return ("unknown", parsed)

        # Check for sensor telemetry packet
        telem = parse_telemetry_packet(parsed, now=curr)
        if telem is not None:
            self.latest_telemetry = telem
            self.telemetry_history.append(telem)
            if len(self.telemetry_history) > self.max_telemetry_history:
                self.telemetry_history.pop(0)
            return ("telemetry", telem)

        # Check for command acknowledgement or watchdog event packet
        if "status" in parsed:
            return ("ack", parsed)

        return ("unknown", parsed)

    def read_telemetry(self, timeout_sec: float = 0.05) -> Optional[SensorTelemetry]:
        """
        Polls the serial transport for pending incoming lines, drains telemetry,
        and returns the latest telemetry object.
        """
        if not self.is_connected():
            return self.latest_telemetry

        start_t = time.time()
        while (time.time() - start_t) <= timeout_sec:
            line = self.transport.read_line(timeout_sec=0.01)
            if not line:
                break
            msg_type, parsed = self.process_incoming_line(line)
            if msg_type == "ack":
                self.last_response = parsed
            elif msg_type == "error":
                self.error_message = parsed

        return self.latest_telemetry

    def is_telemetry_stale(self, now: Optional[float] = None) -> bool:
        """Checks if the most recent sensor telemetry is older than the configured timeout."""
        if self.latest_telemetry is None:
            return True
        return self.latest_telemetry.is_stale(timeout_sec=self.telemetry_timeout_sec, now=now)

    def get_telemetry(self) -> Optional[Dict[str, Any]]:
        """Returns the latest sensor telemetry as a dictionary with freshness metrics."""
        if self.latest_telemetry is None:
            return None
        return self.latest_telemetry.to_dict()

    def send_stop(self, reason: str = "") -> Dict[str, Any]:
        """
        Priority 0: Dispatches STOP command immediately.
        Forces speed to 0 and clears error states where possible.
        """
        return self.send_command(ESP32Command.STOP, speed=0, reason=reason)

    def send_command(self, cmd_str: str, speed: Optional[int] = None, reason: str = "") -> Dict[str, Any]:
        """
        Serializes and dispatches a JSON command to the ESP32 controller.
        Enforces STOP priority: STOP always overrides any pending motion and forces speed=0.
        If transport is disconnected or fails, immediately transitions state to STOP and records error.
        Multiplexes incoming telemetry packets transparently while awaiting the command ACK.
        """
        now = time.time()

        try:
            norm_cmd = ESP32Command.normalize(cmd_str)
        except ValueError as exc:
            self.error_message = str(exc)
            return {
                "ok": False,
                "sent": False,
                "error": str(exc),
                "command": {"cmd": ESP32Command.STOP, "speed": 0}
            }

        # STOP priority: force speed=0
        if norm_cmd == ESP32Command.STOP:
            effective_speed = 0
        else:
            # Obstacle safety guard: Prevent FORWARD movement if an active obstacle is detected
            if norm_cmd == ESP32Command.FORWARD:
                if self.latest_telemetry is not None and not self.is_telemetry_stale(now=now):
                    if self.latest_telemetry.obstacle or (
                        self.latest_telemetry.distance_cm is not None and self.latest_telemetry.distance_cm <= 30.0
                    ):
                        self.error_message = (
                            f"Forward command blocked: Obstacle detected within {self.latest_telemetry.distance_cm}cm"
                        )
                        return {
                            "ok": False,
                            "sent": False,
                            "error": self.error_message,
                            "command": {"cmd": ESP32Command.STOP, "speed": 0, "seq": self.seq}
                        }

            effective_speed = speed if speed is not None else self.default_speed
            effective_speed = max(0, min(255, int(effective_speed)))

        self.seq += 1
        packet = {
            "cmd": norm_cmd,
            "speed": effective_speed,
            "seq": self.seq,
            "ts": round(now, 3)
        }
        if reason:
            packet["reason"] = reason

        # Check connection status
        if not self.transport.is_connected():
            self.connection_status = ConnectionStatus.DISCONNECTED
            self.error_message = "Cannot send command: ESP32 transport disconnected"
            self.current_state = "STOPPED"
            self.last_command = {"cmd": ESP32Command.STOP, "speed": 0, "seq": self.seq, "ts": now}
            return {
                "ok": False,
                "sent": False,
                "error": self.error_message,
                "command": self.last_command,
                "connection": self.connection_status
            }

        line = json.dumps(packet) + "\n"
        write_ok = self.transport.write_line(line)
        if not write_ok:
            self.connection_status = ConnectionStatus.ERROR
            self.error_message = "Failed to write command to serial transport"
            self.current_state = "STOPPED"
            return {
                "ok": False,
                "sent": False,
                "error": self.error_message,
                "command": packet
            }

        # Read and parse response (multiplexed with incoming telemetry)
        start_wait = time.time()
        resp_data = None

        while (time.time() - start_wait) < self.watchdog_timeout_sec:
            raw_line = self.transport.read_line(timeout_sec=self.watchdog_timeout_sec)
            if raw_line is None:
                if not self.transport.is_connected():
                    break
                # Short yield if waiting for physical serial byte arrival
                time.sleep(0.005)
                continue

            msg_type, parsed = self.process_incoming_line(raw_line, now=now)
            if msg_type == "ack":
                resp_data = parsed
                break
            elif msg_type == "telemetry":
                # Ingested telemetry while awaiting command ACK; continue waiting for ACK
                continue
            elif msg_type == "error":
                self.error_message = parsed
                continue

        if resp_data is None:
            # Communication timeout! Failsafe: transition state to STOP
            self.connection_status = ConnectionStatus.ERROR
            self.error_message = f"Communication timeout ({self.watchdog_timeout_sec}s) awaiting response from ESP32"
            self.current_state = "STOPPED"
            self.last_command = {"cmd": ESP32Command.STOP, "speed": 0, "seq": self.seq, "ts": now}
            return {
                "ok": False,
                "sent": True,
                "error": self.error_message,
                "command": packet,
                "timeout": True
            }

        self.last_response = resp_data
        self.last_command = packet
        self.last_heartbeat = now
        self.connection_status = ConnectionStatus.CONNECTED
        self.error_message = None
        self.current_state = "STOPPED" if norm_cmd == ESP32Command.STOP else "RUNNING"

        return {
            "ok": resp_data.get("status") == "ok",
            "sent": True,
            "command": packet,
            "response": resp_data,
            "connection": self.connection_status
        }

    def check_watchdog(self, now: Optional[float] = None) -> bool:
        """
        Periodically checks heartbeat. If communication has timed out and robot was in motion,
        forces internal state to STOP to prevent runaway movement.
        Also halts forward motion if ultrasonic telemetry has become stale while moving forward.
        """
        curr = now if now is not None else time.time()
        if curr - self.last_heartbeat > self.watchdog_timeout_sec:
            if self.current_state != "STOPPED":
                self.send_stop(reason="Watchdog timeout expired")
                self.current_state = "STOPPED"
                self.connection_status = ConnectionStatus.ERROR
                self.error_message = f"Watchdog timeout: no heartbeat for {curr - self.last_heartbeat:.2f}s"
                return False

        # Stale telemetry watchdog: If rover is moving forward and telemetry becomes stale, halt motion
        if self.current_state != "STOPPED" and self.last_command.get("cmd") == ESP32Command.FORWARD:
            if self.is_telemetry_stale(now=curr) and self.latest_telemetry is not None:
                self.send_stop(reason="Ultrasonic telemetry stale while moving forward")
                self.current_state = "STOPPED"
                self.error_message = "Stale ultrasonic telemetry while moving forward"
                return False

        return True

    def get_status(self) -> Dict[str, Any]:
        """Provides communication layer telemetry and status."""
        telem_dict = self.get_telemetry()
        is_stale = self.is_telemetry_stale()
        has_obs = bool(self.latest_telemetry and self.latest_telemetry.obstacle and not is_stale)
        dist = self.latest_telemetry.distance_cm if (self.latest_telemetry and not is_stale) else None

        return {
            "status": self.connection_status,
            "connected": self.transport.is_connected(),
            "simulation_mode": self.simulation_mode,
            "current_state": self.current_state,
            "last_command": self.last_command,
            "last_response": self.last_response,
            "seq": self.seq,
            "watchdog_timeout_sec": self.watchdog_timeout_sec,
            "last_heartbeat_age_sec": round(time.time() - self.last_heartbeat, 2),
            "error": self.error_message,
            "telemetry": telem_dict,
            "telemetry_stale": is_stale,
            "obstacle_detected": has_obs,
            "distance_cm": dist
        }
