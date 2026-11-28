"""Automated tests for the Spam Email Classifier FastAPI endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.app import app


def test_health_endpoint():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model_name"]
        assert 0.9 <= data["accuracy"] <= 1.0


def test_model_info_endpoint():
    with TestClient(app) as client:
        resp = client.get("/api/v1/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["best_model_name"]
        assert len(data["comparison"]) == 5
        assert data["comparison"][0]["is_best"] is True


def test_predict_spam_and_ham():
    with TestClient(app) as client:
        spam_resp = client.post(
            "/api/v1/predict",
            json={
                "subject": "Free $1000 gift card",
                "text": "Congratulations! You won a free $1000 gift card. Click here to claim your prize now!",
            },
        )
        assert spam_resp.status_code == 200
        spam_data = spam_resp.json()
        assert spam_data["label"] == "SPAM"
        assert spam_data["label_id"] == 1
        assert spam_data["spam_probability"] >= 0.5

        ham_resp = client.post(
            "/api/v1/predict",
            json={
                "subject": "Project meeting notes",
                "text": "Hi team, please find attached the meeting notes from yesterday's project sync. Thanks!",
            },
        )
        assert ham_resp.status_code == 200
        ham_data = ham_resp.json()
        assert ham_data["label"] == "HAM"
        assert ham_data["label_id"] == 0
        assert ham_data["ham_probability"] >= 0.5


def test_predict_rejects_blank_text():
    with TestClient(app) as client:
        resp = client.post("/api/v1/predict", json={"text": "   "})
        assert resp.status_code == 422


def test_predict_batch():
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/predict/batch",
            json={
                "emails": [
                    {
                        "text": "URGENT! Buy cheap meds online with 80% discount, click here now!"
                    },
                    {
                        "text": "Thanks for your email, let's meet tomorrow morning to review the schedule."
                    },
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
        assert data["spam_count"] + data["ham_count"] == 2
        assert data["results"][0]["label"] == "SPAM"
        assert data["results"][1]["label"] == "HAM"


def test_demo_ui_served():
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Spam Email Classifier" in resp.text
