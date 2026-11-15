from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


ISSDA_API_BASE_URL = "https://issda.ucd.ie/api"
CER_DATASET_PERSISTENT_ID = "doi:10.7929/ISSDA/BX59EU"


@dataclass(slots=True)
class DataverseFile:
    file_id: int
    filename: str
    label: str
    restricted: bool
    directory_label: str | None
    filesize: int | None


def _headers(api_key: str | None) -> dict[str, str]:
    return {"X-Dataverse-Key": api_key} if api_key else {}


def _extract_files(metadata: dict[str, Any]) -> list[DataverseFile]:
    rows: list[DataverseFile] = []
    latest_version = metadata.get("latestVersion", {})
    for row in latest_version.get("files", []):
        data_file = row.get("dataFile", {})
        file_id = data_file.get("id")
        filename = data_file.get("filename")
        if file_id is None or filename is None:
            continue
        rows.append(
            DataverseFile(
                file_id=int(file_id),
                filename=str(filename),
                label=str(row.get("label") or filename),
                restricted=bool(row.get("restricted")),
                directory_label=row.get("directoryLabel"),
                filesize=int(data_file["filesize"]) if data_file.get("filesize") is not None else None,
            )
        )
    return rows


def fetch_dataset_metadata(
    *,
    client: httpx.Client | None = None,
    persistent_id: str = CER_DATASET_PERSISTENT_ID,
) -> dict[str, Any]:
    owned_client = client is None
    client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        response = client.get(
            f"{ISSDA_API_BASE_URL}/datasets/:persistentId/",
            params={"persistentId": persistent_id},
        )
        response.raise_for_status()
        payload = response.json()
        return payload["data"]
    finally:
        if owned_client:
            client.close()


def _download_file(
    *,
    client: httpx.Client,
    file_id: int,
    target_path: Path,
    api_key: str | None,
) -> str | None:
    with client.stream(
        "GET",
        f"{ISSDA_API_BASE_URL}/access/datafile/{file_id}",
        headers=_headers(api_key),
    ) as response:
        if response.status_code == 403:
            return "forbidden"
        response.raise_for_status()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)
    return None


def download_cer_dataset(
    target_dir: Path,
    *,
    api_key: str | None = None,
    persistent_id: str = CER_DATASET_PERSISTENT_ID,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    owned_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        metadata = fetch_dataset_metadata(client=client, persistent_id=persistent_id)
        files = _extract_files(metadata)
        (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        downloaded_public: list[str] = []
        downloaded_restricted: list[str] = []
        skipped_restricted: list[str] = []
        failed: list[dict[str, Any]] = []

        for file in files:
            subdir = "data" if file.restricted else "documentation"
            target_path = target_dir / subdir / file.filename
            if file.restricted and not api_key:
                skipped_restricted.append(file.filename)
                continue

            error = _download_file(
                client=client,
                file_id=file.file_id,
                target_path=target_path,
                api_key=api_key,
            )
            if error == "forbidden":
                skipped_restricted.append(file.filename)
                continue
            if error is not None:
                failed.append({"filename": file.filename, "error": error})
                continue
            if file.restricted:
                downloaded_restricted.append(file.filename)
            else:
                downloaded_public.append(file.filename)

        return {
            "persistent_id": persistent_id,
            "target_dir": str(target_dir),
            "downloaded_public_files": downloaded_public,
            "downloaded_restricted_files": downloaded_restricted,
            "skipped_restricted_files": skipped_restricted,
            "failed_files": failed,
            "api_key_supplied": bool(api_key),
        }
    finally:
        if owned_client:
            client.close()
