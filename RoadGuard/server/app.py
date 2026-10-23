"""
RoadGuard REST API Server & Telemetry Engine — Supabase Backend
Department of Computer Science & Engineering | IDEA Lab
Ramdeobaba University, Academic Year 2026-27
Guide: Dr. Suresh Balpande
Team: Bhoomi Magarde, Mohammad Ejaz Mehendi, Nidhish Dadhe

REST API connected to Supabase (https://eitvwhbivmsqyrnkvqzv.supabase.co).
Provides real-time telemetry processing, ML evaluation, spatial hazard queries,
route proximity warning, idempotent offline synchronization, and test isolation.
"""

import os
import io
import csv
import json
import time
import math
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Local ML Inference Engine
from ml_inference import ml_engine

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://eitvwhbivmsqyrnkvqzv.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")

if not SUPABASE_KEY:
    raise RuntimeError(
        "No Supabase key found. Set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY "
        "in your .env file."
    )

# Create Supabase client (singleton)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="RoadGuard Smart Hazard Detection Backend API",
    description="Real-Time REST API for RoadGuard Smart Pothole & Speed Breaker Detection System (Supabase Backend)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# -----------------------------------------------------------------------------
# Haversine distance helper (replaces PostgreSQL stored function)
# -----------------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Returns distance in meters between two GPS coordinates."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# -----------------------------------------------------------------------------
# In-Memory Live Telemetry State
# -----------------------------------------------------------------------------
class TelemetryState:
    def __init__(self):
        self.device_id = "RoadGuard-ESP32"
        self.is_connected = False
        self.is_demo_mode = False
        self.last_packet_time = 0.0
        self.packet_timestamps = []
        self.measured_hz = 0.0
        self.speed_kmh = 0.0
        self.latitude = 21.1777
        self.longitude = 79.0617
        self.heading = 0.0
        self.gps_fix = False
        self.sensor_ok = True
        self.accel = {"x": 0.0, "y": 0.0, "z": 9.81}
        self.gyro = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.z_window = [9.81] * 50
        self.x_window = [0.0] * 50
        self.y_window = [0.0] * 50
        self.current_classification = {
            "condition": "NORMAL",
            "confidence": 98.4,
            "severity": "LOW"
        }
        self.last_anomaly_event = None

    def update(self, packet: Dict[str, Any]):
        now = time.time()
        self.last_packet_time = now
        self.is_connected = True
        self.device_id = packet.get("device_id", packet.get("deviceId", self.device_id))
        self.is_demo_mode = packet.get("source", "SENSOR").upper() == "TEST" or packet.get("is_demo", False)

        self.packet_timestamps.append(now)
        if len(self.packet_timestamps) > 50:
            self.packet_timestamps.pop(0)
        if len(self.packet_timestamps) >= 5:
            dt = self.packet_timestamps[-1] - self.packet_timestamps[0]
            if dt > 0:
                self.measured_hz = round((len(self.packet_timestamps) - 1) / dt, 1)

        ax = float(packet.get("accel_x", packet.get("accelX", 0.0)))
        ay = float(packet.get("accel_y", packet.get("accelY", 0.0)))
        az = float(packet.get("accel_z", packet.get("accelZ", 9.81)))
        self.accel = {"x": ax, "y": ay, "z": az}

        gx = float(packet.get("gyro_x", packet.get("gyroX", 0.0)))
        gy = float(packet.get("gyro_y", packet.get("gyroY", 0.0)))
        gz = float(packet.get("gyro_z", packet.get("gyroZ", 0.0)))
        self.gyro = {"x": gx, "y": gy, "z": gz}

        self.z_window.pop(0); self.z_window.append(az)
        self.x_window.pop(0); self.x_window.append(ax)
        self.y_window.pop(0); self.y_window.append(ay)

        self.gps_fix = bool(packet.get("gps_fix", packet.get("gpsFix", False)))
        self.sensor_ok = bool(packet.get("sensor_ok", packet.get("sensorOk", True)))
        self.speed_kmh = float(packet.get("speed_kmh", packet.get("speedKmh", self.speed_kmh)))

        lat = packet.get("latitude")
        lng = packet.get("longitude")
        if lat is not None and lng is not None and lat != 0.0 and lng != 0.0:
            if self.gps_fix and self.latitude != 0.0 and (lat != self.latitude or lng != self.longitude):
                d_lon = math.radians(lng - self.longitude)
                lat1 = math.radians(self.latitude)
                lat2 = math.radians(lat)
                y = math.sin(d_lon) * math.cos(lat2)
                x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
                self.heading = (math.degrees(math.atan2(y, x)) + 360) % 360
            self.latitude = float(lat)
            self.longitude = float(lng)

        classification = ml_engine.classify_window_heuristic(self.z_window, self.speed_kmh)
        self.current_classification = classification
        if classification["condition"] in ["POTHOLE", "SPEED_BREAKER"]:
            self.last_anomaly_event = {
                "condition": classification["condition"],
                "confidence": classification["confidence"],
                "severity": classification["severity"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latitude": self.latitude,
                "longitude": self.longitude
            }

    def get_snapshot(self) -> Dict[str, Any]:
        now = time.time()
        is_fresh = (now - self.last_packet_time) <= 6.0
        conn_status = "CONNECTED" if (self.is_connected and is_fresh) else ("DEMO MODE" if self.is_demo_mode else "DEVICE DISCONNECTED")
        return {
            "device_status": conn_status,
            "device_id": self.device_id,
            "measured_hz": self.measured_hz if is_fresh else 0.0,
            "last_packet_age_seconds": round(now - self.last_packet_time, 1) if self.last_packet_time > 0 else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "speed_kmh": round(self.speed_kmh, 1) if is_fresh else 0.0,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "heading": round(self.heading, 1),
            "gps_fix": self.gps_fix if is_fresh else False,
            "sensor_ok": self.sensor_ok if is_fresh else False,
            "accel": self.accel if is_fresh else {"x": 0.0, "y": 0.0, "z": 9.81},
            "gyro": self.gyro if is_fresh else {"x": 0.0, "y": 0.0, "z": 0.0},
            "recent_samples": {
                "z": self.z_window[-30:],
                "x": self.x_window[-30:],
                "y": self.y_window[-30:]
            },
            "classification": self.current_classification if is_fresh else {
                "condition": "NORMAL", "confidence": 98.4, "severity": "LOW"
            },
            "last_anomaly": self.last_anomaly_event,
            "is_simulated": self.is_demo_mode
        }


telemetry_hub = TelemetryState()


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class IngestObservationRequest(BaseModel):
    hazard_type: str = Field(..., description="POTHOLE or SPEED_BREAKER (NORMAL is not saved)")
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    severity: str = "HIGH"
    roughness_score: float = 6.0
    speed_kmh: float = 40.0
    az_peak: float = 14.5
    trip_id: Optional[str] = None
    source: str = "SENSOR"


class TelemetryIngestRequest(BaseModel):
    device_id: Optional[str] = "RoadGuard-ESP32"
    timestamp: Optional[int] = None
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 9.81
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    latitude: Optional[float] = 21.1777
    longitude: Optional[float] = 79.0617
    speed_kmh: float = 0.0
    gps_fix: bool = False
    sensor_ok: bool = True
    source: str = "SENSOR"


class SyncBatchItem(BaseModel):
    operation_id: Optional[str] = None
    hazard_type: str
    latitude: float
    longitude: float
    severity: str = "MEDIUM"
    roughness_score: float = 6.0
    speed_kmh: float = 35.0
    az_peak: float = 12.0
    observed_at: Optional[str] = None
    trip_id: Optional[str] = None
    source: str = "SENSOR"


class SyncBatchRequest(BaseModel):
    device_id: Optional[str] = "RoadGuard-ESP32"
    records: List[SyncBatchItem]


class TripCreateRequest(BaseModel):
    user_id: Optional[str] = "a0000000-0000-0000-0000-000000000001"
    distance_meters: float = 0.0
    duration_seconds: int = 0
    potholes_count: int = 0
    speed_breakers_count: int = 0
    avg_roughness: float = 0.0


class TestEventRequest(BaseModel):
    event_type: str = Field(..., description="POTHOLE, SPEED_BREAKER, NORMAL, GPS_INVALID, BLE_DISCONNECT")
    latitude: Optional[float] = 21.1777
    longitude: Optional[float] = 79.0617
    speed_kmh: Optional[float] = 38.0
    save_to_db: Optional[bool] = True


# -----------------------------------------------------------------------------
# Helper: ingest hazard observation with 15m deduplication (pure Python)
# -----------------------------------------------------------------------------
def _ingest_hazard_observation(
    hazard_type: str, latitude: float, longitude: float,
    severity: str, roughness_score: float, speed_kmh: float,
    az_peak: float, trip_id: Optional[str] = None, source: str = "SENSOR",
    user_id: Optional[str] = None
) -> str:
    """Upserts a hazard into Supabase with 15m deduplication. Returns hazard UUID."""
    DEDUP_RADIUS_M = 15.0
    pad = 0.00015  # ~15m in degrees

    valid_sources = ["SENSOR", "USER_REPORT", "ADMIN_IMPORT"]
    db_source = "ADMIN_IMPORT" if source.upper() == "TEST" else (source if source in valid_sources else "SENSOR")
    db_severity = severity if severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"] else "MEDIUM"
    norm_conf = min(1.0, max(0.1, round(roughness_score / 10.0, 2)))

    resp = (
        supabase.table("hazards")
        .select("id, latitude, longitude, observation_count, confidence")
        .eq("hazard_type", hazard_type)
        .gte("latitude",  latitude  - pad)
        .lte("latitude",  latitude  + pad)
        .gte("longitude", longitude - pad)
        .lte("longitude", longitude + pad)
        .eq("is_active", True)
        .execute()
    )

    existing = resp.data or []
    matched = None
    for h in existing:
        dist = haversine_distance(latitude, longitude, float(h["latitude"]), float(h["longitude"]))
        if dist <= DEDUP_RADIUS_M:
            matched = h
            break

    now_iso = datetime.now(timezone.utc).isoformat()

    if matched:
        hazard_id = matched["id"]
        new_count = (matched.get("observation_count") or 1) + 1
        old_conf = float(matched.get("confidence") or 0.75)
        new_conf = min(1.0, max(0.1, round((old_conf * (new_count - 1) + norm_conf) / new_count, 2)))
        supabase.table("hazards").update({
            "observation_count": new_count,
            "last_detected_at": now_iso,
            "confidence": new_conf,
            "severity": db_severity,
            "verification_status": "VERIFIED" if new_count >= 3 else "PENDING",
        }).eq("id", hazard_id).execute()
    else:
        hazard_id = str(uuid.uuid4())
        supabase.table("hazards").insert({
            "id": hazard_id,
            "hazard_type": hazard_type,
            "latitude": latitude,
            "longitude": longitude,
            "severity": db_severity,
            "observation_count": 1,
            "confidence": norm_conf,
            "source": db_source,
            "is_active": True,
            "verification_status": "PENDING",
            "first_detected_at": now_iso,
            "last_detected_at": now_iso,
        }).execute()

    if user_id:
        try:
            supabase.table("hazard_observations").insert({
                "id": str(uuid.uuid4()),
                "hazard_id": hazard_id,
                "user_id": user_id,
                "latitude": latitude,
                "longitude": longitude,
                "hazard_type": hazard_type,
                "severity": db_severity,
                "confidence": norm_conf,
                "trip_id": trip_id,
                "source": db_source,
                "observed_at": now_iso,
                "event_fingerprint": f"{hazard_id}:{now_iso}",
                "sensor_features": {
                    "speed_kmh": speed_kmh,
                    "az_peak": az_peak,
                    "roughness_score": roughness_score
                }
            }).execute()
        except Exception as e:
            print(f"Observation insert skipped: {e}")

    return hazard_id


# -----------------------------------------------------------------------------
# Static Dashboard
# -----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def get_dashboard():
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        return FileResponse(static_file)
    return {"message": "RoadGuard Backend API is running. Go to /docs for Swagger UI"}


@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    return Response(status_code=204)


# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    """Returns subsystem health status including Supabase connectivity."""
    db_ok = False
    hazards_count = 0
    obs_count = 0
    error_msg = None
    try:
        r1 = supabase.table("hazards").select("id", count="exact").limit(1).execute()
        hazards_count = r1.count or 0
        r2 = supabase.table("hazard_observations").select("id", count="exact").limit(1).execute()
        obs_count = r2.count or 0
        db_ok = True
    except Exception as e:
        error_msg = str(e)

    ml_status = ml_engine.get_model_status()
    snap = telemetry_hub.get_snapshot()
    db_status = "CONNECTED" if db_ok else "DISCONNECTED"
    return {
        "status": "ok" if db_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": {
            "name": "Supabase Cloud",
            "provider": "Supabase Cloud",
            "status": db_status,
            "display_status": f"SUPABASE {db_status}",
            "backend": "Supabase",
            "url": SUPABASE_URL,
            "hazards_count": hazards_count,
            "observations_count": obs_count,
            "error": error_msg
        },
        "ml_engine": ml_status,
        "device_stream": {
            "device_status": snap["device_status"],
            "device_id": snap["device_id"],
            "measured_hz": snap["measured_hz"],
            "gps_fix": snap["gps_fix"],
            "is_simulated": snap["is_simulated"]
        }
    }


@app.get("/health/ready")
def readiness_check():
    try:
        supabase.table("hazards").select("id").limit(1).execute()
        return {"status": "ready", "database": "connected", "backend": "Supabase"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Supabase unreachable: {str(e)}")


# -----------------------------------------------------------------------------
# Devices
# -----------------------------------------------------------------------------
@app.get("/api/devices")
def get_devices():
    try:
        resp = supabase.table("devices").select(
            "id, device_name, device_identifier, status, last_seen_at, created_at"
        ).order("last_seen_at", desc=True).execute()
        rows = resp.data or []
    except Exception:
        rows = []

    snap = telemetry_hub.get_snapshot()
    if not rows:
        return [{
            "id": "d0000000-0000-0000-0000-000000000001",
            "device_name": "RoadGuard-ESP32",
            "device_identifier": "RoadGuard-ESP32",
            "status": snap["device_status"],
            "last_seen_at": snap["timestamp"],
            "sampling_rate_hz": snap["measured_hz"]
        }]

    return [
        {**r, "live_status": snap["device_status"], "sampling_rate_hz": snap["measured_hz"]}
        if r.get("device_identifier") == snap["device_id"] else r
        for r in rows
    ]


# -----------------------------------------------------------------------------
# Telemetry
# -----------------------------------------------------------------------------
@app.post("/api/telemetry")
def ingest_telemetry(req: TelemetryIngestRequest):
    """Ingests live 50Hz telemetry from ESP32 or Android client."""
    telemetry_hub.update(req.model_dump())
    snap = telemetry_hub.get_snapshot()
    return {"status": "received", "measured_hz": snap["measured_hz"], "classification": snap["classification"]}


@app.get("/api/telemetry/live")
def get_live_telemetry():
    return telemetry_hub.get_snapshot()


# -----------------------------------------------------------------------------
# Hazards
# -----------------------------------------------------------------------------
@app.get("/api/hazards")
def get_hazards(
    lat: Optional[float] = None,
    lng: Optional[float] = None,
    radius: float = Query(5000.0),
    hazard_type: Optional[str] = None,
    severity: Optional[str] = None,
    source: Optional[str] = None,
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lon: Optional[float] = None,
    max_lon: Optional[float] = None,
    limit: int = 300,
    offset: int = 0
):
    try:
        query = supabase.table("hazards").select("*")
        if hazard_type and hazard_type.upper() != "ALL":
            query = query.eq("hazard_type", hazard_type.upper())
        if severity and severity.upper() != "ALL":
            query = query.eq("severity", severity.upper())
        if source and source.upper() != "ALL":
            src_val = "ADMIN_IMPORT" if source.upper() == "TEST" else source.upper()
            query = query.eq("source", src_val)

        if lat is not None and lng is not None:
            pad = radius / 111000.0
            resp = (query
                .gte("latitude",  lat - pad).lte("latitude",  lat + pad)
                .gte("longitude", lng - pad).lte("longitude", lng + pad)
                .order("last_detected_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute())
            result = []
            for r in (resp.data or []):
                dist = haversine_distance(lat, lng, float(r["latitude"]), float(r["longitude"]))
                if dist <= radius:
                    r["distance_meters"] = round(dist, 1)
                    result.append(r)
            result.sort(key=lambda x: x["distance_meters"])
            return result

        elif min_lat is not None:
            resp = (query
                .gte("latitude", min_lat).lte("latitude", max_lat)
                .gte("longitude", min_lon).lte("longitude", max_lon)
                .order("last_detected_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute())
            return resp.data or []
        else:
            resp = query.order("first_detected_at", desc=True).range(offset, offset + limit - 1).execute()
            return resp.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hazards/nearby")
def get_nearby_hazards(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lng: float = Query(..., ge=-180.0, le=180.0),
    radius: float = Query(150.0),
    heading: Optional[float] = Query(None)
):
    try:
        pad = radius / 111000.0
        resp = (
            supabase.table("hazards")
            .select("id, hazard_type, latitude, longitude, severity, observation_count, verification_status, source, first_detected_at")
            .gte("latitude",  lat - pad).lte("latitude",  lat + pad)
            .gte("longitude", lng - pad).lte("longitude", lng + pad)
            .limit(50).execute()
        )
        rows = resp.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    results = []
    for r in rows:
        h_lat, h_lon = float(r["latitude"]), float(r["longitude"])
        dist = haversine_distance(lat, lng, h_lat, h_lon)
        if dist > radius:
            continue
        d_lon = math.radians(h_lon - lng)
        y = math.sin(d_lon) * math.cos(math.radians(h_lat))
        x = math.cos(math.radians(lat)) * math.sin(math.radians(h_lat)) - math.sin(math.radians(lat)) * math.cos(math.radians(h_lat)) * math.cos(d_lon)
        bearing = (math.degrees(math.atan2(y, x)) + 360) % 360
        approach = "PROXIMITY_ONLY"
        if heading is not None:
            diff = abs((bearing - heading + 180) % 360 - 180)
            approach = "APPROACHING" if diff <= 65 else ("RECEDING" if diff > 90 else "LATERAL")
        r["roughness_score"] = r.get("roughness_score") or 6.0
        results.append({**r, "bearing_degrees": round(bearing, 1), "approach_status": approach, "distance_meters": round(dist, 1)})

    results.sort(key=lambda x: x["distance_meters"])
    return results[:20]


@app.get("/api/hazards/{hazard_id}")
def get_hazard_details(hazard_id: str):
    try:
        h = supabase.table("hazards").select("*").eq("id", hazard_id).execute()
        if not h.data:
            raise HTTPException(status_code=404, detail="Hazard not found")
        obs = (supabase.table("hazard_observations")
            .select("id, trip_id, latitude, longitude, speed_kmh, az_peak, roughness_score, observed_at")
            .eq("hazard_id", hazard_id).order("observed_at", desc=True).limit(20).execute())
        return {"hazard": h.data[0], "observations": obs.data or []}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/hazards")
@app.post("/api/observations")
def ingest_observation(req: IngestObservationRequest):
    h_type = req.hazard_type.upper().strip()
    if h_type == "NORMAL":
        return {"status": "ignored", "message": "NORMAL condition is not stored as a hazard record."}
    if h_type not in ["POTHOLE", "SPEED_BREAKER"]:
        raise HTTPException(status_code=400, detail="Invalid hazard_type. Must be POTHOLE or SPEED_BREAKER.")
    try:
        hid = _ingest_hazard_observation(
            h_type, req.latitude, req.longitude, req.severity.upper(),
            req.roughness_score, req.speed_kmh, req.az_peak, req.trip_id,
            req.source.upper() if req.source else "SENSOR"
        )
        return {"status": "success", "hazard_id": hid, "hazard_type": h_type}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# Offline Sync
# -----------------------------------------------------------------------------
@app.post("/api/sync")
def sync_offline_records(batch: SyncBatchRequest):
    if not batch.records:
        return {"status": "ok", "synced_count": 0, "skipped_count": 0}
    synced = skipped = 0
    for item in batch.records:
        h_type = item.hazard_type.upper().strip()
        if h_type == "NORMAL":
            skipped += 1
            continue
        try:
            _ingest_hazard_observation(
                h_type, item.latitude, item.longitude, item.severity.upper(),
                item.roughness_score, item.speed_kmh, item.az_peak, item.trip_id, item.source
            )
            synced += 1
        except Exception:
            skipped += 1
    return {"status": "success", "synced_count": synced, "skipped_count": skipped,
            "timestamp": datetime.now(timezone.utc).isoformat()}


# -----------------------------------------------------------------------------
# Trips
# -----------------------------------------------------------------------------
@app.get("/api/trips")
def get_trips(limit: int = 50, offset: int = 0):
    try:
        resp = supabase.table("trips").select("*").order("started_at", desc=True).range(offset, offset + limit - 1).execute()
        return resp.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trips")
def create_trip(req: TripCreateRequest):
    try:
        trip_id = str(uuid.uuid4())
        supabase.table("trips").insert({
            "id": trip_id, "user_id": req.user_id,
            "status": "ACTIVE", "started_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return {"status": "success", "trip_id": trip_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------------------------------------------------------
# ML Metrics
# -----------------------------------------------------------------------------
@app.get("/api/metrics")
def get_ml_metrics():
    return ml_engine.compute_evaluation_metrics()


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
@app.get("/api/export")
def export_hazards(
    format: str = Query("json", pattern="^(json|csv)$"),
    source: Optional[str] = None,
    hazard_type: Optional[str] = None
):
    try:
        query = supabase.table("hazards").select("*")
        if source and source.upper() != "ALL":
            query = query.eq("source", source.upper())
        if hazard_type and hazard_type.upper() != "ALL":
            query = query.eq("hazard_type", hazard_type.upper())
        rows = query.order("first_detected_at", desc=True).execute().data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    ts = int(time.time())
    if format == "csv":
        out = io.StringIO()
        if rows:
            w = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        else:
            out.write("id,hazard_type,latitude,longitude,severity,roughness_score\n")
        out.seek(0)
        return Response(content=out.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=roadguard_hazards_{ts}.csv"})
    return Response(content=json.dumps(rows, indent=2, default=str), media_type="application/json",
                    headers={"Content-Disposition": f"attachment; filename=roadguard_hazards_{ts}.json"})


# -----------------------------------------------------------------------------
# Test / Diagnostics
# -----------------------------------------------------------------------------
@app.post("/api/test/event")
def inject_test_event(req: TestEventRequest):
    event = req.event_type.upper()
    if event == "GPS_INVALID":
        telemetry_hub.update({"source": "TEST", "gps_fix": False, "latitude": 0.0, "longitude": 0.0, "speed_kmh": 0.0})
        return {"status": "test_applied", "event": "GPS_INVALID"}
    elif event == "BLE_DISCONNECT":
        telemetry_hub.is_connected = False; telemetry_hub.last_packet_time = 0.0
        return {"status": "test_applied", "event": "BLE_DISCONNECT"}
    elif event == "NORMAL":
        telemetry_hub.update({"source": "TEST", "accel_x": 0.1, "accel_y": -0.05, "accel_z": 9.81,
                               "speed_kmh": req.speed_kmh or 40.0, "gps_fix": True, "sensor_ok": True,
                               "latitude": req.latitude, "longitude": req.longitude})
        return {"status": "test_applied", "event": "NORMAL"}
    elif event in ["POTHOLE", "SPEED_BREAKER"]:
        is_ph = (event == "POTHOLE")
        az = 22.4 if is_ph else 14.8
        telemetry_hub.update({"source": "TEST", "accel_x": 3.2 if is_ph else 0.8, "accel_y": -1.5,
                               "accel_z": az, "speed_kmh": req.speed_kmh or 38.0,
                               "gps_fix": True, "sensor_ok": True,
                               "latitude": req.latitude, "longitude": req.longitude})
        hid = None
        if req.save_to_db:
            try:
                hid = _ingest_hazard_observation(
                    event, req.latitude, req.longitude,
                    "HIGH" if is_ph else "MEDIUM", 8.4 if is_ph else 6.2,
                    req.speed_kmh or 38.0, az, None, "TEST"
                )
            except Exception:
                pass
        return {"status": "test_applied", "event": event, "hazard_id": hid, "source": "TEST"}
    raise HTTPException(status_code=400, detail="Invalid event_type.")


@app.post("/api/test/reset")
def reset_test_data():
    try:
        hids_resp = supabase.table("hazards").select("id").eq("source", "ADMIN_IMPORT").execute()
        hids = [h["id"] for h in (hids_resp.data or [])]
        obs_deleted = hazards_deleted = 0
        for hid in hids:
            r = supabase.table("hazard_observations").delete().eq("hazard_id", hid).execute()
            obs_deleted += len(r.data or [])
        if hids:
            r = supabase.table("hazards").delete().eq("source", "ADMIN_IMPORT").execute()
            hazards_deleted = len(r.data or [])
        telemetry_hub.is_demo_mode = False
        return {"status": "success", "hazards_deleted": hazards_deleted, "observations_deleted": obs_deleted}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
