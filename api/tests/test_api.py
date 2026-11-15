from __future__ import annotations


def test_login_and_me_for_admin(client):
    login = client.post("/auth/login", json={"email": "admin@demo.local", "password": "Admin123!"})
    assert login.status_code == 200
    tokens = login.json()
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert me.status_code == 200
    assert me.json()["role"] in {"admin", "ADMIN"}


def test_activation_returns_user_and_tokens_and_allows_me(client):
    activate = client.post(
        "/auth/activate",
        json={"token": "activate-household-0002", "password": "Household#2026"},
    )
    assert activate.status_code == 200
    payload = activate.json()
    assert payload["user"]["email"] == "household-0002@demo.local"
    assert payload["tokens"]["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {payload['tokens']['access_token']}"})
    assert me.status_code == 200
    assert me.json()["is_active"] is True


def test_refresh_rotates_refresh_token_and_logout_revokes_session(client):
    login = client.post("/auth/login", json={"email": "admin@demo.local", "password": "Admin123!"})
    assert login.status_code == 200
    tokens = login.json()

    refreshed = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refreshed.status_code == 200
    next_tokens = refreshed.json()
    assert next_tokens["refresh_token"] != tokens["refresh_token"]

    reused = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert reused.status_code == 401

    logout = client.post("/auth/logout", json={"refresh_token": next_tokens["refresh_token"]})
    assert logout.status_code == 200

    after_logout = client.post("/auth/refresh", json={"refresh_token": next_tokens["refresh_token"]})
    assert after_logout.status_code == 401


def test_activation_rejects_weak_password(client):
    activate = client.post(
        "/auth/activate",
        json={"token": "activate-household-0002", "password": "weakpass"},
    )
    assert activate.status_code == 422


def test_household_history_is_scoped_to_authenticated_user(client):
    login = client.post("/auth/login", json={"email": "household-0001@demo.local", "password": "Household123!"})
    tokens = login.json()
    response = client.get("/household/history?household_id=999", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert response.status_code == 200
    body = response.json()
    assert body["household_code"] == "HOUSEHOLD-0001"
    assert len(body["series"]) == 1


def test_admin_can_create_training_run_and_validation_flow(client):
    login = client.post("/auth/login", json={"email": "admin@demo.local", "password": "Admin123!"})
    token = login.json()["access_token"]

    create_run = client.post(
        "/admin/runs",
        json={
            "config": {
                "dataset_version_id": 1,
                "client_sampling_rate": 1.0,
                "rounds": 3,
                "local_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "secure_aggregation_enabled": True,
                "secure_aggregation_degree": 2,
                "secure_aggregation_mask_scale": 4.0,
                "dp_enabled": True,
                "noise_multiplier": 1.1,
                "clipping_norm": 1.0,
            }
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert create_run.status_code == 200
    run = create_run.json()
    assert run["status"] in {"pending", "PENDING"}

    models = client.get("/admin/models", headers={"Authorization": f"Bearer {token}"})
    assert models.status_code == 200
    model_id = models.json()[0]["id"]

    validate = client.post(
        f"/admin/models/{model_id}/validate",
        json={"notes": "validated in test"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert validate.status_code == 200

    approve = client.post(
        f"/admin/models/{model_id}/approve",
        json={"notes": "approved in test"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert approve.status_code == 200

    publish = client.post(
        "/admin/models/publish",
        json={"model_version_id": model_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert publish.status_code == 200
    assert publish.json()["job_type"] in {"publish_model", "PUBLISH_MODEL"}


def test_admin_can_queue_cer_dataset_import(client):
    login = client.post("/auth/login", json={"email": "admin@demo.local", "password": "Admin123!"})
    token = login.json()["access_token"]

    response = client.post(
        "/admin/datasets/import",
        json={
            "source_type": "cer",
            "profile": "cer-dev",
            "sample_households": 8,
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()["payload"]
    assert payload["source_type"] == "cer"
    assert payload["profile"] == "cer-dev"
    assert payload["sample_households"] == 8


def test_admin_can_queue_smart_meter_dataset_import(client):
    login = client.post("/auth/login", json={"email": "admin@demo.local", "password": "Admin123!"})
    token = login.json()["access_token"]

    response = client.post(
        "/admin/datasets/import",
        json={
            "source_type": "smart-meters",
            "profile": "smart-meters",
            "dataset_path": "/workspace/data/artifacts/datasets/smart-meters",
            "sample_households": 16,
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    payload = response.json()["payload"]
    assert payload["source_type"] == "smart-meters"
    assert payload["profile"] == "smart-meters"
    assert payload["dataset_path"] == "/workspace/data/artifacts/datasets/smart-meters"
    assert payload["sample_households"] == 16


def test_dataset_import_rejects_invalid_source_profile_pair(client):
    login = client.post("/auth/login", json={"email": "admin@demo.local", "password": "Admin123!"})
    token = login.json()["access_token"]

    response = client.post(
        "/admin/datasets/import",
        json={
            "source_type": "fixture",
            "profile": "cer-dev",
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422

    smart_meter_response = client.post(
        "/admin/datasets/import",
        json={
            "source_type": "smart-meters",
            "profile": "cer-dev",
        },
        headers={"Authorization": f"Bearer {token}"},
    )

    assert smart_meter_response.status_code == 422
