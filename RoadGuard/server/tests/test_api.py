"""
RoadGuard Automated Test Suite
Validates API endpoints, database deduplication, 3-class classification handling,
spatial proximity calculations, ML metrics, export, and test isolation.
"""

import sys
import os
import pytest  # noqa: F401
from fastapi.testclient import TestClient

# Ensure server module is in path
SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

try:
    from app import app
except ImportError:
    from RoadGuard.server.app import app  # type: ignore

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "database" in data
    assert data["database"]["status"] == "CONNECTED"
    assert "ml_engine" in data
    assert data["ml_engine"]["model_configured"] is True
    assert "NORMAL" in data["ml_engine"]["classes"]
    assert "POTHOLE" in data["ml_engine"]["classes"]
    assert "SPEED_BREAKER" in data["ml_engine"]["classes"]


def test_readiness_endpoint():
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_devices_endpoint():
    response = client.get("/api/devices")
    assert response.status_code == 200
    devices = response.json()
    assert isinstance(devices, list)
    assert len(devices) > 0
    assert "RoadGuard-ESP32" in [d["device_name"] for d in devices]


def test_telemetry_ingest_and_live_polling():
    packet = {
        "device_id": "RoadGuard-ESP32-TEST",
        "accel_x": 0.15,
        "accel_y": -0.05,
        "accel_z": 9.81,
        "speed_kmh": 40.0,
        "latitude": 21.1777,
        "longitude": 79.0617,
        "gps_fix": True,
        "sensor_ok": True,
        "source": "TEST"
    }
    post_res = client.post("/api/telemetry", json=packet)
    assert post_res.status_code == 200
    assert post_res.json()["status"] == "received"

    get_res = client.get("/api/telemetry/live")
    assert get_res.status_code == 200
    snap = get_res.json()
    assert snap["device_id"] == "RoadGuard-ESP32-TEST"
    assert snap["gps_fix"] is True
    assert "classification" in snap


def test_normal_road_never_saved_as_hazard():
    payload = {
        "hazard_type": "NORMAL",
        "latitude": 21.1777,
        "longitude": 79.0617,
        "severity": "LOW",
        "source": "TEST"
    }
    response = client.post("/api/hazards", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ignored"
    assert "NORMAL condition is not stored" in data["message"]


def test_hazard_ingest_and_deduplication():
    # Ingest test pothole
    test_lat = 21.1850
    test_lng = 79.0700
    payload = {
        "hazard_type": "POTHOLE",
        "latitude": test_lat,
        "longitude": test_lng,
        "severity": "HIGH",
        "roughness_score": 8.0,
        "speed_kmh": 35.0,
        "az_peak": 22.0,
        "source": "TEST"
    }
    res1 = client.post("/api/hazards", json=payload)
    assert res1.status_code == 200
    hazard_id1 = res1.json()["hazard_id"]

    # Ingest within 10 meters (should deduplicate to same hazard ID)
    payload2 = {
        "hazard_type": "POTHOLE",
        "latitude": test_lat + 0.00005,  # ~5.5 meters offset
        "longitude": test_lng + 0.00005,
        "severity": "HIGH",
        "roughness_score": 8.5,
        "speed_kmh": 36.0,
        "az_peak": 23.0,
        "source": "TEST"
    }
    res2 = client.post("/api/hazards", json=payload2)
    assert res2.status_code == 200
    hazard_id2 = res2.json()["hazard_id"]
    assert hazard_id1 == hazard_id2, "Nearby observations within 15m must deduplicate to same hazard!"

    # Clean up test records
    client.post("/api/test/reset")


def test_nearby_hazards_and_approach_direction():
    # Check nearby endpoint
    response = client.get("/api/hazards/nearby?lat=21.1777&lng=79.0617&radius=1000&heading=55")
    assert response.status_code == 200
    nearby = response.json()
    assert isinstance(nearby, list)
    if nearby:
        first = nearby[0]
        assert "distance_meters" in first
        assert "bearing_degrees" in first
        assert "approach_status" in first
        assert first["approach_status"] in ["APPROACHING", "RECEDING", "LATERAL", "PROXIMITY_ONLY"]


def test_ml_metrics_endpoint():
    response = client.get("/api/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["status"] == "EVALUATED"
    assert "overall_accuracy" in metrics
    assert "per_class_metrics" in metrics
    assert "confusion_matrix" in metrics
    assert "NORMAL" in metrics["per_class_metrics"]
    assert "POTHOLE" in metrics["per_class_metrics"]
    assert "SPEED_BREAKER" in metrics["per_class_metrics"]


def test_export_endpoints():
    # JSON Export
    res_json = client.get("/api/export?format=json")
    assert res_json.status_code == 200
    assert res_json.headers["content-type"] == "application/json"

    # CSV Export
    res_csv = client.get("/api/export?format=csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    assert "latitude" in res_csv.text and "hazard_type" in res_csv.text


def test_offline_sync_batch():
    batch = {
        "device_id": "RoadGuard-Android-App",
        "records": [
            {
                "hazard_type": "SPEED_BREAKER",
                "latitude": 21.1755,
                "longitude": 79.0595,
                "severity": "MEDIUM",
                "roughness_score": 5.5,
                "speed_kmh": 25.0,
                "az_peak": 14.0,
                "source": "TEST"
            },
            {
                "hazard_type": "NORMAL",
                "latitude": 21.1756,
                "longitude": 79.0596,
                "source": "TEST"
            }
        ]
    }
    response = client.post("/api/sync", json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["synced_count"] == 1
    assert data["skipped_count"] == 1  # NORMAL is skipped

    # Clean up test data
    client.post("/api/test/reset")


def test_test_data_isolation_and_reset():
    client.post("/api/test/event", json={
        "event_type": "POTHOLE",
        "latitude": 21.2850,
        "longitude": 79.1850,
        "save_to_db": True
    })

    # Verify test pothole exists
    hazards_before = client.get("/api/hazards?source=TEST").json()
    assert len(hazards_before) >= 1

    # Reset test data
    reset_res = client.post("/api/test/reset")
    assert reset_res.status_code == 200

    # Verify test hazards are gone
    hazards_after = client.get("/api/hazards?source=TEST").json()
    assert len(hazards_after) == 0

    # Verify real/sensor hazards are preserved
    real_hazards = client.get("/api/hazards?source=SENSOR").json()
    assert len(real_hazards) > 0


def test_health_database_provider_contract():
    res = client.get("/health")
    assert res.status_code == 200
    db = res.json()["database"]
    assert db["provider"] == "Supabase Cloud"
    assert db["status"] == "CONNECTED"
    assert db["display_status"] == "SUPABASE CONNECTED"
    assert db["error"] is None
    assert db["hazards_count"] >= 3


def test_map_filters_backend():
    # Filter potholes
    potholes = client.get("/api/hazards?hazard_type=POTHOLE").json()
    assert all(h["hazard_type"] == "POTHOLE" for h in potholes)

    # Filter speed breakers
    breakers = client.get("/api/hazards?hazard_type=SPEED_BREAKER").json()
    assert all(h["hazard_type"] == "SPEED_BREAKER" for h in breakers)

    # Filter real sensor data
    sensor_data = client.get("/api/hazards?source=SENSOR").json()
    assert all(h["source"] == "SENSOR" for h in sensor_data)


def test_gps_loss_and_recovery():
    # Inject GPS loss
    res = client.post("/api/test/event", json={"event_type": "GPS_INVALID"})
    assert res.status_code == 200
    snap = client.get("/api/telemetry/live").json()
    assert snap["gps_fix"] is False

    # Recover with valid GPS telemetry
    client.post("/api/telemetry", json={
        "device_id": "RoadGuard-ESP32-TEST",
        "accel_x": 0.0, "accel_y": 0.0, "accel_z": 9.81,
        "latitude": 21.1777, "longitude": 79.0617,
        "speed_kmh": 40.0, "gps_fix": True, "sensor_ok": True,
        "source": "TEST"
    })
    snap2 = client.get("/api/telemetry/live").json()
    assert snap2["gps_fix"] is True
    assert snap2["latitude"] == 21.1777


def test_empty_and_failed_requests():
    # 404 on non-existent hazard
    res404 = client.get("/api/hazards/00000000-0000-0000-0000-000000000000")
    assert res404.status_code == 404

    # 400 on invalid hazard type
    res_bad = client.post("/api/hazards", json={
        "hazard_type": "UNKNOWN_CRATER",
        "latitude": 21.1777,
        "longitude": 79.0617
    })
    assert res_bad.status_code == 400

    # 422 on invalid coordinates
    res_bad_coords = client.post("/api/hazards", json={
        "hazard_type": "POTHOLE",
        "latitude": 999.0,
        "longitude": 79.0617
    })
    assert res_bad_coords.status_code == 422


def test_proximity_calculation_and_straight_line():
    # Vehicle at 21.1777, 79.0617 moving north (heading = 0)
    res = client.get("/api/hazards/nearby?lat=21.1777&lng=79.0617&radius=1000&heading=0")
    assert res.status_code == 200
    nearby = res.json()
    assert isinstance(nearby, list)
    for h in nearby:
        assert "distance_meters" in h
        assert "bearing_degrees" in h
        assert "approach_status" in h
        assert h["distance_meters"] <= 1000.0


def test_dashboard_html_contains_required_elements():
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    # Required controls
    assert "TEST SIREN" in html
    assert "TEST VOICE ALERT" in html
    assert "VOICE ALERTS" in html
    assert "STOP ALERT" in html
    assert "AUDIO READY" in html
    assert "Simulate Approaching Hazard" in html
    # Spoken warning text
    assert "Warning! Pothole ahead. Reduce speed." in html
    assert "Warning! Speed breaker ahead. Reduce speed." in html
    # Filters
    assert "Potholes Only" in html
    assert "Breakers Only" in html
    assert "Real Detections" in html
    assert "Test Data" in html
    # Straight-line distance
    assert "STRAIGHT-LINE DISTANCE" in html

