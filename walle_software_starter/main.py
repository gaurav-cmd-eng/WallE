import os
import time
import threading
from typing import Optional, Dict, Any, List

import cv2
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from tracker import CentroidTracker, TrackedObject
except ImportError:
    from walle_software_starter.tracker import CentroidTracker, TrackedObject

try:
    from autonomous import AutonomousDecisionEngine, Decision, AutonomousState
except ImportError:
    from walle_software_starter.autonomous import AutonomousDecisionEngine, Decision, AutonomousState

try:
    from follow_controller import FollowController, FollowMode, FollowState, FollowDecision
    from person_tracker import PersonObservation
    from tag_tracker import RadioSource
    from person_detector import PersonDetectorManager, DetectorStatus
    from tag_simulation import DeterministicTagSimulator
    from person_simulation import DeterministicPersonSimulator
    from sim_arena import SimulationArena
    from esp32_comm import ESP32CommunicationManager, ESP32Command, SerialTransport, MockSerialTransport, ConnectionStatus
except ImportError:
    from walle_software_starter.follow_controller import FollowController, FollowMode, FollowState, FollowDecision
    from walle_software_starter.person_tracker import PersonObservation
    from walle_software_starter.tag_tracker import RadioSource
    from walle_software_starter.person_detector import PersonDetectorManager, DetectorStatus
    from walle_software_starter.tag_simulation import DeterministicTagSimulator
    from walle_software_starter.person_simulation import DeterministicPersonSimulator
    from walle_software_starter.sim_arena import SimulationArena
    from walle_software_starter.esp32_comm import ESP32CommunicationManager, ESP32Command, SerialTransport, MockSerialTransport, ConnectionStatus

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ESP32_URL = os.getenv("ESP32_URL", "").strip()  # Example: http://192.168.1.50
ESP32_PORT = os.getenv("ESP32_PORT", "").strip()  # Example: /dev/ttyUSB0 or COM3
ESP32_BAUDRATE = int(os.getenv("ESP32_BAUDRATE", "115200"))
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

# Configurable detection parameters
MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", "100"))
MIN_OBJECT_AREA = int(os.getenv("MIN_OBJECT_AREA", "500"))
MERGE_DISTANCE_THRESHOLD = int(os.getenv("MERGE_DISTANCE_THRESHOLD", "25"))

# Configurable tracking parameters
MAX_DISAPPEARED_FRAMES = int(os.getenv("MAX_DISAPPEARED_FRAMES", "15"))
MAX_TRACKING_DISTANCE = float(os.getenv("MAX_TRACKING_DISTANCE", "80.0"))
MOVEMENT_THRESHOLD = float(os.getenv("MOVEMENT_THRESHOLD", "3.0"))

# Configurable person detection inference stride
PERSON_INFERENCE_INTERVAL = int(os.getenv("PERSON_INFERENCE_INTERVAL", "2"))

tracker = CentroidTracker(
    max_disappeared=MAX_DISAPPEARED_FRAMES,
    max_distance=MAX_TRACKING_DISTANCE,
    movement_threshold=MOVEMENT_THRESHOLD
)

autonomous_engine = AutonomousDecisionEngine()
follow_controller = FollowController()

if ESP32_PORT:
    serial_transport = SerialTransport(port=ESP32_PORT, baudrate=ESP32_BAUDRATE)
    conn_ok = serial_transport.connect()
    if conn_ok:
        esp32_comm = ESP32CommunicationManager(transport=serial_transport, simulation_mode=False)
    else:
        esp32_comm = ESP32CommunicationManager(simulation_mode=True)
        esp32_comm.error_message = f"Failed to connect to serial port {ESP32_PORT}"
else:
    esp32_comm = ESP32CommunicationManager(simulation_mode=True)

follow_controller.command_dispatcher = lambda direction, speed: esp32_comm.send_command(direction, speed)
person_detector = PersonDetectorManager()
person_detector.set_camera_checker(lambda: state.get("camera") == "connected")

# Stage 6: Interactive 2D Simulation Arena instance
sim_arena = SimulationArena(follow_controller=follow_controller, esp32_comm=esp32_comm)


def sync_esp32_telemetry():
    """
    Polls fresh sensor telemetry from the ESP32 communication manager
    and feeds it directly into the follow controller safety system.
    """
    esp32_comm.read_telemetry(timeout_sec=0.005)
    t = esp32_comm.get_telemetry()
    is_stale = esp32_comm.is_telemetry_stale()
    if t is not None:
        follow_controller.update_sensor_telemetry(
            distance_cm=t.get("distance_cm"),
            obstacle=t.get("obstacle", False),
            is_valid=t.get("is_valid", True),
            is_stale=is_stale
        )
    elif esp32_comm.latest_telemetry is not None:
        follow_controller.update_sensor_telemetry(
            distance_cm=esp32_comm.latest_telemetry.distance_cm,
            obstacle=esp32_comm.latest_telemetry.obstacle,
            is_valid=False,
            is_stale=True
        )

    # Active motion safety: If rover is moving forward and obstacle is detected or telemetry becomes stale
    if state.get("drive") == "forward":
        is_stale_now = esp32_comm.is_telemetry_stale()
        obs = follow_controller.obstacle_detected or (esp32_comm.latest_telemetry and esp32_comm.latest_telemetry.obstacle)
        if obs or (esp32_comm.latest_telemetry is not None and is_stale_now):
            send_to_esp32("/drive", {"direction": "stop", "speed": 0})
            state["drive"] = "stopped"
            state["last_command"] = "Safety halt: Stale telemetry or obstacle while moving forward"


follow_controller.sensor_telemetry_source = sync_esp32_telemetry


def get_sensor_status() -> Dict[str, Any]:
    t = esp32_comm.get_telemetry()
    is_stale = esp32_comm.is_telemetry_stale()
    imu_dict = t.get("imu") if (t and isinstance(t.get("imu"), dict)) else {}
    return {
        "connected": esp32_comm.is_connected(),
        "telemetry_received": t is not None,
        "is_valid": t.get("is_valid", False) if t else False,
        "is_stale": is_stale,
        "age_sec": t.get("age_sec") if t else None,
        "distance_cm": t.get("distance_cm") if t else None,
        "obstacle": bool(t.get("obstacle", False)) if t else False,
        "imu": {
            "ax": imu_dict.get("ax", 0.0),
            "ay": imu_dict.get("ay", 0.0),
            "az": imu_dict.get("az", 0.0),
            "gx": imu_dict.get("gx", 0.0),
            "gy": imu_dict.get("gy", 0.0),
            "gz": imu_dict.get("gz", 0.0),
        },
        "ultrasonic_error": "ok" if (t and t.get("is_valid")) else ("stale" if is_stale else "no_data"),
        "imu_error": "ok" if (t and imu_dict) else "no_data",
        "error": esp32_comm.error_message
    }

app = FastAPI(title="WALL-E Control Center")
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")

state = {
    "mode": "esp32" if (ESP32_URL or (ESP32_PORT and esp32_comm.is_connected())) else "simulation",
    "drive": "stopped",
    "emergency_stop_latched": False,
    "last_command": "System ready",
    "camera": "starting",
    "object": {
        "detected": False,
        "id": None,
        "color": "NONE",
        "direction": "NO TARGET",
        "movement": "NONE",
        "x": 0,
        "y": 0,
        "area": 0,
    },
    "objects": [],
    "object_count": 0,
}
camera = None
camera_lock = threading.Lock()


class DriveCommand(BaseModel):
    direction: str = Field(pattern="^(forward|backward|left|right|stop)$")
    speed: int = Field(default=150, ge=0, le=255)


class AutonomousEnableCommand(BaseModel):
    enabled: bool


class AutonomousTargetCommand(BaseModel):
    target_id: Optional[int] = None


class FollowModeCommand(BaseModel):
    mode: str = Field(pattern="^(MANUAL|PERSON|POCKET_TAG)$")


class FollowEnableCommand(BaseModel):
    enabled: bool


class FollowPersonTargetCommand(BaseModel):
    person_id: Optional[int] = None


class FollowTagTargetCommand(BaseModel):
    tag_id: Optional[str] = None
    source: Optional[str] = None


class FollowTagSourceCommand(BaseModel):
    source: str = Field(pattern="^(?i)(UWB|BLE)$")


class SimTagCommand(BaseModel):
    tag_id: str = "TAG_01"
    source: str = "UWB"
    distance_m: Optional[float] = 1.5
    signal_quality: float = 95.0
    rssi: Optional[float] = -65.0
    direction: Optional[str] = "CENTER"
    connected: bool = True
    battery_pct: Optional[float] = 100.0


class TagPairCommand(BaseModel):
    tag_id: str
    name: Optional[str] = ""
    action: str = Field(default="pair", pattern="^(pair|unpair)$")


class TagDirectionCommand(BaseModel):
    direction: Optional[str] = Field(default=None, pattern="^(LEFT|RIGHT|CENTER|UNKNOWN|NONE)?$")
    tag_id: Optional[str] = None
    ttl_sec: Optional[float] = Field(default=0.5, ge=0.01, le=10.0)
    source: str = "EXTERNAL"


class TagScenarioCommand(BaseModel):
    scenario: str = "all"
    seed: Optional[int] = 42


class SimPersonCommand(BaseModel):
    person_id: int = 1
    bbox: list = [200, 100, 150, 300]
    confidence: float = 0.95
    direction: Optional[str] = "CENTER"
    features: Optional[list] = None
    is_valid: bool = True


class DetectorModeCommand(BaseModel):
    mode: str = Field(pattern="^(REAL|SIMULATION|AUTO)$")


class BenchmarkCommand(BaseModel):
    iterations: int = Field(default=10, ge=1, le=50)


class SensorConfigCommand(BaseModel):
    obstacle_threshold_cm: Optional[float] = Field(default=None, gt=0.0)
    require_ultrasonic_for_forward: Optional[bool] = None


class PersonScenarioCommand(BaseModel):
    scenario: str = "all"
    seed: Optional[int] = 42


class SimPeopleBatchCommand(BaseModel):
    people: List[SimPersonCommand]


class SimSensorCommand(BaseModel):
    distance_cm: Optional[float] = 85.0
    obstacle: bool = False
    ax: float = 0.0
    ay: float = 0.0
    az: float = 9.81
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    is_valid: bool = True
    is_stale: bool = False


class SimESP32DisconnectCommand(BaseModel):
    disconnected: bool


class ArenaMovePersonCommand(BaseModel):
    person_id: int = 1
    dx: float = 0.0
    dy: float = 0.0


class ArenaAddPersonCommand(BaseModel):
    person_id: Optional[int] = None
    x: float = 5.0
    y: float = 4.0


class ArenaAddObstacleCommand(BaseModel):
    obs_id: str
    x: float
    y: float
    radius: float = 0.4


class ArenaRepositionObstacleCommand(BaseModel):
    obs_id: str
    x: float
    y: float
    radius: Optional[float] = None


class ArenaRepositionPersonCommand(BaseModel):
    distance_m: Optional[float] = 2.5


class ArenaSensorFailCommand(BaseModel):
    fail: bool


class ArenaAvoidanceToggleCommand(BaseModel):
    enabled: Optional[bool] = None


def send_to_esp32(path: str, params: dict):
    # Dispatch through serial/mock communication manager
    direction = params.get("direction", "stop")
    speed = params.get("speed", 0)
    comm_res = esp32_comm.send_command(direction, speed)

    if not ESP32_URL:
        return {"sent": False, "mode": "simulation", "esp32_comm": comm_res}
    try:
        response = httpx.get(f"{ESP32_URL.rstrip('/')}{path}", params=params, timeout=1.5)
        response.raise_for_status()
        return {"sent": True, "mode": "esp32", "esp32_comm": comm_res}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach ESP32: {exc}")


def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(CAMERA_INDEX)
            if camera.isOpened():
                state["camera"] = "connected"
            else:
                state["camera"] = "not found"
                camera.release()
                camera = None
        return camera


@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(APP_DIR, "static", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/status")
def status():
    sync_esp32_telemetry()
    return {
        **state,
        "emergency_stop_latched": follow_controller.emergency_stop_latched,
        "autonomous": autonomous_engine.get_status(),
        "follow": {
            **follow_controller.get_status(),
            "person_detector": person_detector.get_health()
        },
        "sensors": get_sensor_status(),
        "esp32_comm": esp32_comm.get_status(),
        "esp32_configured": bool(ESP32_URL or ESP32_PORT)
    }


@app.get("/api/esp32/status")
def get_esp32_status():
    return {"ok": True, "esp32_comm": esp32_comm.get_status()}


@app.post("/api/drive")
def drive(command: DriveCommand):
    sync_esp32_telemetry()
    if follow_controller.emergency_stop_latched and command.direction != "stop":
        raise HTTPException(
            status_code=400,
            detail="Cannot drive: Emergency STOP latch is active. Call reset endpoint first."
        )

    if command.direction == "forward":
        clear, reason = follow_controller.is_forward_path_clear()
        if not clear:
            raise HTTPException(
                status_code=400,
                detail=f"Forward movement blocked by obstacle safety guard: {reason}"
            )

    result = send_to_esp32("/drive", {"direction": command.direction, "speed": command.speed})
    state["drive"] = "stopped" if command.direction == "stop" else command.direction
    state["last_command"] = f"Drive: {command.direction}"
    return {"ok": True, **result}


@app.post("/api/stop")
def stop():
    if autonomous_engine.enabled:
        autonomous_engine.manual_override()
    follow_controller.emergency_stop()
    esp32_comm.send_stop(reason="Emergency STOP")
    result = send_to_esp32("/drive", {"direction": "stop", "speed": 0})
    state["drive"] = "stopped"
    state["emergency_stop_latched"] = True
    state["last_command"] = "Emergency stop latched"
    return {"ok": True, **result}


@app.post("/api/follow/emergency_stop/reset")
@app.post("/api/safety/reset")
def reset_safety_latch():
    follow_controller.reset_emergency_stop()
    state["emergency_stop_latched"] = False
    state["last_command"] = "Emergency stop latch reset"
    return {
        "ok": True,
        "emergency_stop_latched": False,
        "follow": follow_controller.get_status()
    }


@app.post("/api/autonomous/enable")
def set_autonomous_enable(command: AutonomousEnableCommand):
    autonomous_engine.set_enabled(command.enabled)
    state["last_command"] = f"Autonomous: {'ENABLED' if command.enabled else 'DISABLED'}"
    return {"ok": True, "autonomous": autonomous_engine.get_status()}


@app.post("/api/autonomous/target")
def set_autonomous_target(command: AutonomousTargetCommand):
    autonomous_engine.set_target_id(command.target_id)
    target_str = f"#{command.target_id}" if command.target_id is not None else "AUTO"
    state["last_command"] = f"Target set: {target_str}"
    return {"ok": True, "autonomous": autonomous_engine.get_status()}


@app.get("/api/autonomous/decision")
def get_autonomous_decision():
    return {"ok": True, "autonomous": autonomous_engine.get_status()}


@app.post("/api/autonomous/override")
def autonomous_override():
    autonomous_engine.manual_override()
    state["drive"] = "stopped"
    state["last_command"] = "Manual override: STOP"
    return {"ok": True, "autonomous": autonomous_engine.get_status()}


@app.get("/api/autonomous/commands")
def get_autonomous_commands():
    return {
        "ok": True,
        "last_command": autonomous_engine.command_interface.last_command,
        "history": autonomous_engine.command_interface.command_history,
        "simulation_mode": True
    }


@app.post("/api/follow/mode")
def set_follow_mode(cmd: FollowModeCommand):
    try:
        follow_controller.set_mode(cmd.mode)
        state["last_command"] = f"Follow mode: {cmd.mode}"
        return {"ok": True, "follow": follow_controller.get_status()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/follow/enable")
def set_follow_enable(cmd: FollowEnableCommand):
    if follow_controller.emergency_stop_latched and cmd.enabled:
        raise HTTPException(
            status_code=400,
            detail="Cannot enable following: Emergency STOP latch is active. Call reset endpoint first."
        )
    follow_controller.set_enabled(cmd.enabled)
    state["last_command"] = f"Follow: {'ENABLED' if cmd.enabled else 'DISABLED'}"
    return {"ok": True, "follow": follow_controller.get_status()}


@app.post("/api/follow/target/person")
def set_follow_person_target(cmd: FollowPersonTargetCommand):
    follow_controller.select_person_target(cmd.person_id)
    target_str = f"Person #{cmd.person_id}" if cmd.person_id is not None else "AUTO"
    state["last_command"] = f"Follow target: {target_str}"
    return {"ok": True, "follow": follow_controller.get_status()}


@app.post("/api/follow/target/tag")
def set_follow_tag_target(cmd: FollowTagTargetCommand):
    if cmd.source and cmd.source.upper() not in ("UWB", "BLE"):
        raise HTTPException(status_code=400, detail="Invalid source. Expected UWB or BLE.")
    follow_controller.select_tag_target(cmd.tag_id, cmd.source)
    target_str = f"Tag {cmd.tag_id or 'AUTO'} ({cmd.source or follow_controller.tag_tracker.active_source})"
    state["last_command"] = f"Follow target: {target_str}"
    return {"ok": True, "follow": follow_controller.get_status()}


@app.post("/api/follow/tag/source")
def set_follow_tag_source(cmd: FollowTagSourceCommand):
    follow_controller.tag_tracker.set_source(cmd.source)
    state["last_command"] = f"Tag radio source: {cmd.source.upper()}"
    return {"ok": True, "follow": follow_controller.get_status()}


@app.get("/api/follow/person/status")
def get_follow_person_status():
    return {
        "ok": True,
        "person_tracker": follow_controller.person_tracker.get_status(),
        "detector": person_detector.get_health()
    }


@app.get("/api/person/detected")
def get_detected_people():
    people = follow_controller.person_tracker.get_tracked_people()
    target = follow_controller.person_tracker.get_selected_target()
    return {
        "ok": True,
        "people": [p.to_dict() for p in people],
        "count": len(people),
        "target": target.to_dict() if target else None,
        "selected_person_id": follow_controller.person_tracker.selected_person_id,
        "reacquisition_status": follow_controller.person_tracker.reacquisition_status,
        "detector": person_detector.get_health()
    }


@app.post("/api/person/target/auto")
def set_person_target_auto():
    follow_controller.person_tracker.set_auto_target()
    state["last_command"] = "Person target: AUTO"
    return {"ok": True, "follow": follow_controller.get_status()}


@app.post("/api/person/target/clear")
def clear_person_target():
    follow_controller.person_tracker.clear_target()
    state["last_command"] = "Person target: CLEARED"
    return {"ok": True, "follow": follow_controller.get_status()}


@app.get("/api/person/detector/health")
def get_person_detector_health():
    return {"ok": True, "detector": person_detector.get_health()}


@app.post("/api/person/detector/mode")
def set_person_detector_mode(cmd: DetectorModeCommand):
    try:
        person_detector.set_mode(cmd.mode)
        state["last_command"] = f"Person detector mode: {cmd.mode}"
        return {"ok": True, "detector": person_detector.get_health()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/person/detector/benchmark")
def run_person_detector_benchmark(cmd: BenchmarkCommand):
    result = person_detector.benchmark(iterations=cmd.iterations)
    return {"ok": True, "benchmark": result}


@app.get("/api/follow/tag/status")
def get_follow_tag_status():
    status = follow_controller.tag_tracker.get_status()
    status["following_config"] = {
        "target_distance_m": follow_controller.target_distance_m,
        "approach_threshold_m": follow_controller.approach_threshold_m,
        "hold_min_m": follow_controller.hold_min_m,
        "hold_max_m": follow_controller.hold_max_m,
        "min_safe_distance_m": follow_controller.min_safe_distance_m,
    }
    return {"ok": True, "tag_tracker": status}


@app.post("/api/follow/tag/pair")
def pair_or_unpair_tag(cmd: TagPairCommand):
    if cmd.action.lower() == "unpair":
        res = follow_controller.tag_tracker.unpair_tag(cmd.tag_id)
        state["last_command"] = f"Unpaired tag: {cmd.tag_id}"
        return {"ok": True, "unpaired": res, "tags": follow_controller.tag_tracker.get_paired_tags()}
    else:
        info = follow_controller.tag_tracker.pair_tag(cmd.tag_id, name=cmd.name or "")
        state["last_command"] = f"Paired tag: {cmd.tag_id}"
        return {"ok": True, "tag": info.to_dict(), "tags": follow_controller.tag_tracker.get_paired_tags()}


@app.get("/api/follow/tag/tags")
def list_follow_tags():
    return {
        "ok": True,
        "tags": follow_controller.tag_tracker.get_paired_tags(),
        "selected_tag_id": follow_controller.tag_tracker.selected_tag_id,
        "enforce_pairing": follow_controller.tag_tracker.enforce_pairing
    }


@app.post("/api/follow/tag/direction")
def set_follow_tag_direction(cmd: TagDirectionCommand):
    dir_val = cmd.direction.upper() if cmd.direction and cmd.direction.upper() in ("LEFT", "RIGHT", "CENTER") else None
    if dir_val is None:
        follow_controller.tag_tracker.clear_external_direction(cmd.tag_id)
        state["last_command"] = f"Tag direction cleared ({cmd.source})"
    else:
        follow_controller.tag_tracker.set_external_direction(
            dir_val,
            tag_id=cmd.tag_id,
            ttl_sec=cmd.ttl_sec or 0.5
        )
        state["last_command"] = f"Tag direction: {dir_val} ({cmd.source})"
    if follow_controller.enabled:
        follow_controller.evaluate()
    return {"ok": True, "follow": follow_controller.get_status()}


@app.post("/api/follow/tag/direction/clear")
def clear_follow_tag_direction(tag_id: Optional[str] = None):
    follow_controller.tag_tracker.clear_external_direction(tag_id)
    state["last_command"] = "Tag direction cleared"
    if follow_controller.enabled:
        follow_controller.evaluate()
    return {"ok": True, "follow": follow_controller.get_status()}


@app.post("/api/follow/tag/scenario")
def run_tag_scenario(cmd: TagScenarioCommand):
    simulator = DeterministicTagSimulator(seed=cmd.seed or 42)
    scen_name = cmd.scenario.lower()
    if scen_name in ("all", "*"):
        report = simulator.run_all_scenarios()
        return {"ok": True, "report": report}

    method_name = f"scenario_{scen_name}"
    if not hasattr(simulator, method_name):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{cmd.scenario}'. Available: 'all' or scenario methods (e.g. 'uwb_1_no_direction', 'ble_1_near', etc.)"
        )
    fn = getattr(simulator, method_name)
    result = fn()
    return {"ok": True, "scenario": result}


@app.get("/api/follow/status")
def get_follow_status():
    sync_esp32_telemetry()
    if follow_controller.enabled:
        follow_controller.evaluate()
    return {
        "ok": True,
        "follow": follow_controller.get_status(),
        "sensors": get_sensor_status()
    }


@app.get("/api/sensors/telemetry")
def get_sensors_telemetry():
    sync_esp32_telemetry()
    return {
        "ok": True,
        "sensors": get_sensor_status(),
        "follow_safety": {
            "obstacle_detected": follow_controller.obstacle_detected,
            "simulated_obstacle": follow_controller.simulated_obstacle,
            "sensor_obstacle": follow_controller.sensor_obstacle,
            "ultrasonic_distance_cm": follow_controller.ultrasonic_distance_cm,
            "ultrasonic_valid": follow_controller.ultrasonic_valid,
            "ultrasonic_stale": follow_controller.ultrasonic_stale,
            "obstacle_threshold_cm": follow_controller.ultrasonic_obstacle_threshold_cm,
            "require_ultrasonic_for_forward": follow_controller.require_ultrasonic_for_forward,
        }
    }


@app.post("/api/sensors/config")
def set_sensor_config(cmd: SensorConfigCommand):
    if cmd.obstacle_threshold_cm is not None:
        follow_controller.ultrasonic_obstacle_threshold_cm = float(cmd.obstacle_threshold_cm)
    if cmd.require_ultrasonic_for_forward is not None:
        follow_controller.require_ultrasonic_for_forward = bool(cmd.require_ultrasonic_for_forward)
    return {
        "ok": True,
        "obstacle_threshold_cm": follow_controller.ultrasonic_obstacle_threshold_cm,
        "require_ultrasonic_for_forward": follow_controller.require_ultrasonic_for_forward
    }


@app.post("/api/follow/obstacle")
def set_follow_obstacle(detected: bool):
    follow_controller.trigger_obstacle(detected)
    if follow_controller.enabled:
        follow_controller.evaluate()
    return {"ok": True, "follow": follow_controller.get_status()}


@app.post("/api/follow/sim/tag")
def simulate_tag_reading(cmd: SimTagCommand):
    if cmd.source.upper() == "UWB":
        dist = cmd.distance_m if cmd.distance_m is not None else 1.5
        reading = follow_controller.tag_tracker.uwb_tracker.record_measurement(
            tag_id=cmd.tag_id,
            distance_m=dist,
            signal_quality=cmd.signal_quality,
            direction=cmd.direction,
            connected=cmd.connected,
            battery_pct=cmd.battery_pct
        )
    else:
        rssi = cmd.rssi if cmd.rssi is not None else -65.0
        reading = follow_controller.tag_tracker.ble_tracker.record_measurement(
            tag_id=cmd.tag_id,
            rssi=rssi,
            direction=cmd.direction,
            connected=cmd.connected,
            battery_pct=cmd.battery_pct
        )
    if follow_controller.enabled:
        follow_controller.evaluate()
    return {"ok": True, "reading": reading.to_dict(), "follow": follow_controller.get_status()}


@app.post("/api/follow/sim/person")
def simulate_person_detection(cmd: SimPersonCommand):
    obs = PersonObservation(
        person_id=cmd.person_id,
        bbox=tuple(cmd.bbox),
        confidence=cmd.confidence,
        direction=cmd.direction,
        features=cmd.features,
        is_valid=cmd.is_valid
    )
    follow_controller.person_tracker.update([obs])
    if follow_controller.enabled:
        follow_controller.evaluate()
    return {"ok": True, "target": obs.to_dict(), "follow": follow_controller.get_status()}


@app.post("/api/follow/sim/people")
def simulate_multiple_people(cmd: SimPeopleBatchCommand):
    observations = []
    for p in cmd.people:
        obs = PersonObservation(
            person_id=p.person_id,
            bbox=tuple(p.bbox),
            confidence=p.confidence,
            direction=p.direction,
            features=p.features,
            is_valid=p.is_valid
        )
        observations.append(obs)
    follow_controller.person_tracker.update(observations)
    if follow_controller.enabled:
        follow_controller.evaluate()
    return {
        "ok": True,
        "count": len(observations),
        "people": [o.to_dict() for o in observations],
        "follow": follow_controller.get_status()
    }


@app.post("/api/follow/sim/person/clear")
def clear_simulated_people():
    # Feeds empty observation list to simulate all people leaving frame
    for _ in range(6):
        follow_controller.person_tracker.update([])
    if follow_controller.enabled:
        follow_controller.evaluate()
    return {
        "ok": True,
        "cleared": True,
        "follow": follow_controller.get_status()
    }


@app.post("/api/follow/sim/person/preset")
def set_sim_person_preset(preset: str):
    pr = preset.upper()
    if pr == "LEFT":
        obs = PersonObservation(person_id=1, bbox=(60, 80, 130, 280), confidence=0.94, direction="LEFT")
        follow_controller.person_tracker.update([obs], frame_width=640)
    elif pr == "RIGHT":
        obs = PersonObservation(person_id=1, bbox=(490, 80, 130, 280), confidence=0.93, direction="RIGHT")
        follow_controller.person_tracker.update([obs], frame_width=640)
    elif pr == "CENTER":
        obs = PersonObservation(person_id=1, bbox=(260, 80, 130, 280), confidence=0.95, direction="CENTER")
        follow_controller.person_tracker.update([obs], frame_width=640)
    elif pr == "MULTIPLE":
        p1 = PersonObservation(person_id=1, bbox=(50, 80, 120, 260), confidence=0.91, direction="LEFT")
        p2 = PersonObservation(person_id=2, bbox=(260, 70, 130, 280), confidence=0.95, direction="CENTER")
        p3 = PersonObservation(person_id=3, bbox=(480, 85, 115, 255), confidence=0.89, direction="RIGHT")
        follow_controller.person_tracker.update([p1, p2, p3], frame_width=640)
    elif pr in ("CLEAR", "NONE", "LOST"):
        for _ in range(6):
            follow_controller.person_tracker.update([], frame_width=640)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown preset '{preset}'. Available: LEFT, CENTER, RIGHT, MULTIPLE, CLEAR")

    if follow_controller.enabled:
        follow_controller.evaluate()
    return {
        "ok": True,
        "preset": pr,
        "tracked": [p.to_dict() for p in follow_controller.person_tracker.get_tracked_people()],
        "follow": follow_controller.get_status()
    }


@app.post("/api/follow/person/scenario")
def run_person_scenario(cmd: PersonScenarioCommand):
    simulator = DeterministicPersonSimulator(seed=cmd.seed or 42)
    scen_name = cmd.scenario.lower()
    if scen_name in ("all", "*"):
        report = simulator.run_all_scenarios()
        return {"ok": True, "report": report}

    method_name = f"scenario_{scen_name}" if not scen_name.startswith("scenario_") else scen_name
    if not hasattr(simulator, method_name):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario '{cmd.scenario}'. Available: 'all' or scenario methods (e.g. 'scenario_1_multiple_people_detected')"
        )
    fn = getattr(simulator, method_name)
    result = fn()
    return {"ok": True, "scenario": result}


@app.post("/api/sensors/sim")
def simulate_sensor_reading(cmd: SimSensorCommand):
    follow_controller.update_sensor_telemetry(
        distance_cm=cmd.distance_cm,
        obstacle=cmd.obstacle,
        is_valid=cmd.is_valid,
        is_stale=cmd.is_stale
    )
    if isinstance(esp32_comm.transport, MockSerialTransport):
        esp32_comm.transport.queue_telemetry(
            distance_cm=cmd.distance_cm,
            obstacle=cmd.obstacle,
            imu={
                "ax": cmd.ax, "ay": cmd.ay, "az": cmd.az,
                "gx": cmd.gx, "gy": cmd.gy, "gz": cmd.gz
            }
        )
        esp32_comm.read_telemetry(timeout_sec=0.01)
    if follow_controller.enabled:
        follow_controller.evaluate()
    return {
        "ok": True,
        "sensors": get_sensor_status(),
        "follow": follow_controller.get_status()
    }


@app.post("/api/esp32/simulate_disconnect")
def simulate_esp32_disconnect(cmd: SimESP32DisconnectCommand):
    if isinstance(esp32_comm.transport, MockSerialTransport):
        esp32_comm.transport.simulate_disconnect = bool(cmd.disconnected)
        if cmd.disconnected:
            esp32_comm.connection_status = ConnectionStatus.DISCONNECTED
            esp32_comm.current_state = "STOPPED"
            esp32_comm.error_message = "Simulated ESP32 disconnect"
            state["drive"] = "stopped"
        else:
            esp32_comm.connection_status = ConnectionStatus.CONNECTED
            esp32_comm.error_message = None
    return {
        "ok": True,
        "disconnected": cmd.disconnected,
        "esp32_comm": esp32_comm.get_status()
    }


# ============================================================================
# STAGE 6: INTERACTIVE 2D SIMULATION ARENA ENDPOINTS
# ============================================================================
@app.get("/api/arena/state")
def get_arena_state():
    return {"ok": True, "state": sim_arena.get_state()}


@app.post("/api/arena/step")
def step_arena(dt: Optional[float] = None):
    new_state = sim_arena.step(dt=dt)
    return {"ok": True, "state": new_state}


@app.post("/api/arena/reset")
def reset_arena():
    sim_arena.reset()
    return {"ok": True, "state": sim_arena.get_state()}


@app.post("/api/arena/person/move")
def move_arena_person(cmd: ArenaMovePersonCommand):
    sim_arena.move_person(cmd.person_id, cmd.dx, cmd.dy)
    # Automatically execute a simulation step on manual keyboard movement
    new_state = sim_arena.step()
    return {"ok": True, "state": new_state}


@app.post("/api/arena/person/add")
def add_arena_person(cmd: ArenaAddPersonCommand):
    pid = sim_arena.add_person(person_id=cmd.person_id, x=cmd.x, y=cmd.y)
    return {"ok": True, "person_id": pid, "state": sim_arena.get_state()}


@app.delete("/api/arena/person/{person_id}")
def remove_arena_person(person_id: int):
    sim_arena.remove_person(person_id)
    new_state = sim_arena.step()
    return {"ok": True, "removed": person_id, "state": new_state}


@app.post("/api/arena/target/{person_id}")
def set_arena_target(person_id: int):
    sim_arena.selected_target_id = person_id
    for p in sim_arena.people.values():
        p.is_target = (p.id == person_id)
    if follow_controller:
        follow_controller.select_person_target(person_id)
    new_state = sim_arena.step()
    return {"ok": True, "target_id": person_id, "state": new_state}


@app.post("/api/arena/obstacle/add")
def add_arena_obstacle(cmd: ArenaAddObstacleCommand):
    sim_arena.add_obstacle(cmd.obs_id, cmd.x, cmd.y, cmd.radius)
    new_state = sim_arena.step()
    return {"ok": True, "obstacle_id": cmd.obs_id, "state": new_state}


@app.delete("/api/arena/obstacle/{obs_id}")
def remove_arena_obstacle(obs_id: str):
    sim_arena.remove_obstacle(obs_id)
    new_state = sim_arena.step()
    return {"ok": True, "removed": obs_id, "state": new_state}


@app.post("/api/arena/obstacles/clear")
def clear_arena_obstacles():
    sim_arena.clear_obstacles()
    new_state = sim_arena.step()
    return {"ok": True, "state": new_state}


@app.post("/api/arena/obstacle/reposition")
def reposition_arena_obstacle(cmd: ArenaRepositionObstacleCommand):
    sim_arena.reposition_obstacle(cmd.obs_id, cmd.x, cmd.y, cmd.radius)
    new_state = sim_arena.step()
    return {"ok": True, "obstacle_id": cmd.obs_id, "state": new_state}


@app.post("/api/arena/person/reposition")
def reposition_arena_person(cmd: Optional[ArenaRepositionPersonCommand] = None):
    dist = cmd.distance_m if cmd and cmd.distance_m else 2.5
    res = sim_arena.reposition_person_ahead(distance_m=dist)
    new_state = sim_arena.step()
    return {"ok": True, "repositioned_to": res, "state": new_state}


@app.post("/api/arena/reacquire")
def reacquire_arena_target():
    res = sim_arena.reacquire_target()
    new_state = sim_arena.step()
    return {"ok": res.get("ok", True), "info": res, "state": new_state}


@app.post("/api/arena/sensor/fail")
def set_arena_sensor_fail(cmd: ArenaSensorFailCommand):
    sim_arena.simulate_sensor_fail = cmd.fail
    new_state = sim_arena.step()
    return {"ok": True, "sensor_failed": cmd.fail, "state": new_state}


@app.post("/api/arena/avoidance/toggle")
def toggle_arena_avoidance(cmd: Optional[ArenaAvoidanceToggleCommand] = None):
    if cmd and cmd.enabled is not None:
        sim_arena.auto_avoidance_enabled = cmd.enabled
    else:
        sim_arena.auto_avoidance_enabled = not sim_arena.auto_avoidance_enabled
    new_state = sim_arena.step()
    return {"ok": True, "auto_avoidance_enabled": sim_arena.auto_avoidance_enabled, "state": new_state}


@app.post("/api/arena/search/start")
def start_arena_target_search():
    res = sim_arena.start_360_search()
    new_state = sim_arena.step()
    return {"ok": res.get("ok", True), "info": res, "state": new_state}


@app.post("/api/arena/search/cancel")
def cancel_arena_target_search():
    res = sim_arena.cancel_target_search()
    new_state = sim_arena.step()
    return {"ok": res.get("ok", True), "info": res, "state": new_state}




def bbox_distance(b1, b2):
    """Calculate minimum Euclidean distance between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    # Horizontal gap (0 if overlapping in x)
    dx = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
    # Vertical gap (0 if overlapping in y)
    dy = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))

    return (dx * dx + dy * dy) ** 0.5


def merge_contours_by_proximity(contours,
                                min_contour_area=None,
                                min_object_area=None,
                                merge_distance=None):
    """
    Filter small noise contours and merge nearby contours belonging to the same
    physical object using bounding-box proximity grouping (Disjoint Set Union).
    """
    if min_contour_area is None:
        min_contour_area = MIN_CONTOUR_AREA
    if min_object_area is None:
        min_object_area = MIN_OBJECT_AREA
    if merge_distance is None:
        merge_distance = MERGE_DISTANCE_THRESHOLD

    # Filter out small noise contours
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_contour_area:
            candidates.append((c, cv2.boundingRect(c), area))

    if not candidates:
        return []

    n = len(candidates)
    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # Group candidate bounding boxes that are within merge_distance
    for i in range(n):
        for j in range(i + 1, n):
            if bbox_distance(candidates[i][1], candidates[j][1]) <= merge_distance:
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(candidates[i])

    merged_objects = []
    for group in groups.values():
        total_area = sum(item[2] for item in group)
        if total_area < min_object_area:
            continue

        x_min = min(item[1][0] for item in group)
        y_min = min(item[1][1] for item in group)
        x_max = max(item[1][0] + item[1][2] for item in group)
        y_max = max(item[1][1] + item[1][3] for item in group)

        w = x_max - x_min
        h = y_max - y_min

        merged_objects.append({
            "bbox": (int(x_min), int(y_min), int(w), int(h)),
            "area": int(total_area)
        })

    return merged_objects


def detect_colored_objects(frame,
                           min_contour_area=None,
                           min_object_area=None,
                           merge_distance=None,
                           tracker_instance=None):
    """
    Detects coloured objects in a BGR frame across 6 supported colours:
    RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE.
    Merges nearby same-colour contours to prevent multiple detections of a single object.
    Associates detections across consecutive frames using centroid tracking for persistent IDs and movement.
    Annotates the frame with bounding boxes, center points, labels (with ID & movement) and object count.
    Returns:
        tracked_list (list): List of tracked object dicts (id, color, x, y, direction, movement, area)
        frame (np.ndarray): Annotated BGR frame
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_height, frame_width = frame.shape[:2]

    color_ranges = {
        "RED": [
            ((0, 100, 100), (10, 255, 255)),
            ((170, 100, 100), (180, 255, 255))
        ],
        "ORANGE": [
            ((11, 100, 100), (24, 255, 255))
        ],
        "YELLOW": [
            ((25, 100, 100), (34, 255, 255))
        ],
        "GREEN": [
            ((35, 70, 70), (85, 255, 255))
        ],
        "BLUE": [
            ((90, 70, 70), (130, 255, 255))
        ],
        "PURPLE": [
            ((131, 70, 70), (160, 255, 255))
        ]
    }

    raw_detections = []
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for color_name, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            part = cv2.inRange(hsv, tuple(lower), tuple(upper))
            mask = part if mask is None else cv2.bitwise_or(mask, part)

        # Morphological closing to seal micro-holes within solid regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, morph_kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        merged_objects = merge_contours_by_proximity(
            contours,
            min_contour_area=min_contour_area,
            min_object_area=min_object_area,
            merge_distance=merge_distance
        )

        for obj in merged_objects:
            x, y, w, h = obj["bbox"]
            center_x = x + w // 2
            center_y = y + h // 2

            if center_x < frame_width / 3:
                direction = "LEFT"
            elif center_x > frame_width * 2 / 3:
                direction = "RIGHT"
            else:
                direction = "CENTER"

            raw_detections.append({
                "color": color_name,
                "x": center_x,
                "y": center_y,
                "bbox": (x, y, w, h),
                "direction": direction,
                "area": obj["area"]
            })

    # Associate with centroid tracker across consecutive frames
    active_tracker = tracker_instance if tracker_instance is not None else tracker
    tracked_list = active_tracker.update(raw_detections)

    for obj in tracked_list:
        x, y, w, h = obj["bbox"]
        center_x = obj["x"]
        center_y = obj["y"]
        color_name = obj["color"]
        obj_id = obj["id"]
        movement = obj.get("movement", "STATIONARY")

        # Draw bounding box around merged object
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Mark object centre
        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (255, 0, 0),
            -1
        )

        # Display persistent ID, colour, coordinates and movement
        label = f"ID {obj_id}: {color_name} ({center_x}, {center_y})"
        if movement != "STATIONARY":
            label += f" [{movement}]"

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Display total object count
    cv2.putText(
        frame,
        f"Objects: {len(tracked_list)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    return tracked_list, frame


def mjpeg_frames():
    frame_idx = 0
    while True:
        cap = get_camera()

        if cap is None:
            time.sleep(1)
            continue

        ok, frame = cap.read()

        if not ok:
            state["camera"] = "camera read error"
            time.sleep(0.1)
            continue

        state["camera"] = "connected"
        frame_idx += 1

        # 1. Existing six-colour object detection
        tracked_list, frame = detect_colored_objects(frame)

        # 2. Real-time person detection & tracking
        # Use stride to optimize CPU usage on Raspberry Pi 4
        is_det_frame = (frame_idx % PERSON_INFERENCE_INTERVAL == 0)
        if is_det_frame:
            raw_people = person_detector.detect(frame)
            tracked_people = follow_controller.person_tracker.update(
                raw_people,
                frame_width=frame.shape[1],
                is_detection_frame=True
            )
        else:
            tracked_people = follow_controller.person_tracker.update(
                [],
                frame_width=frame.shape[1],
                is_detection_frame=False
            )

        # Annotate detected people on camera frame
        active_target = follow_controller.person_tracker.get_selected_target()
        active_target_id = active_target.id if active_target else None

        for p in tracked_people:
            x, y, w, h = p.bbox
            cx, cy = p.center
            conf_pct = int(p.confidence * 100)
            is_target = (p.id == active_target_id)

            # Gold box for active target, Cyan for other detected people
            box_color = (0, 215, 255) if is_target else (255, 191, 0)
            thickness = 3 if is_target else 2

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, thickness)
            cv2.circle(frame, (cx, cy), 4, box_color, -1)

            label = f"Person #{p.id} ({conf_pct}%) [{p.direction}]"
            if is_target:
                label = f"★ TARGET {label}"

            cv2.putText(
                frame,
                label,
                (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

        # People count & detector performance overlay
        det_health = person_detector.get_health()
        stats_text = f"People: {len(tracked_people)} | Det: {det_health['status']}"
        if det_health.get("fps"):
            stats_text += f" ({det_health['inference_time_ms']:.1f}ms / {det_health['fps']:.1f}fps)"

        cv2.putText(
            frame,
            stats_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        # Autonomous decision engine evaluation
        if autonomous_engine.enabled:
            autonomous_engine.evaluate(tracked_list, frame_width=frame.shape[1])

        # Dual-mode follow controller evaluation
        if follow_controller.enabled:
            follow_controller.evaluate()

        # Update object information
        if tracked_list:
            state["objects"] = tracked_list
            state["object_count"] = len(tracked_list)

            # Keep the largest object as the primary target
            primary = max(tracked_list, key=lambda obj: obj["area"])
            state["object"] = {
                "detected": True,
                **primary
            }
        else:
            state["objects"] = []
            state["object_count"] = 0
            state["object"] = {
                "detected": False,
                "id": None,
                "color": "NONE",
                "direction": "NO TARGET",
                "movement": "NONE",
                "x": 0,
                "y": 0,
                "area": 0
            }


        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 75]
        )

        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + encoded.tobytes()
                + b"\r\n"
            )



@app.get("/video_feed")
def video_feed():
    return StreamingResponse(mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("shutdown")
def shutdown_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
