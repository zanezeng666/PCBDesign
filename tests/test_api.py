from __future__ import annotations

from fastapi.testclient import TestClient

import battery_designer.app as app_module
from battery_designer.storage import ProjectStore


def test_health_and_project_preview(common_spec, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "store", ProjectStore(tmp_path / "projects"))
    client = TestClient(app_module.app)
    assert client.get("/").status_code == 200
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    created = client.post("/api/projects", json=common_spec.model_dump(mode="json"))
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]
    assert created.json()["port_topology"] == "common"

    preview = client.post(f"/api/projects/{project_id}/preview")
    assert preview.status_code == 200, preview.text
    assert preview.json()["stage"] == "preview_ready"
    artifact = client.get(f"/api/projects/{project_id}/artifacts/output/preview/mechanical_front.svg")
    assert artifact.status_code == 200
    assert artifact.text.startswith("<svg")


def test_candidate_production_gate(common_spec, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "store", ProjectStore(tmp_path / "projects"))
    client = TestClient(app_module.app)
    project_id = client.post("/api/projects", json=common_spec.model_dump(mode="json")).json()["id"]
    response = client.post(f"/api/projects/{project_id}/manufacturing")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CANDIDATE_APPROVAL_REQUIRED"
