from __future__ import annotations

import json

import httpx
import respx

from trainer.data.issda import download_cer_dataset


def _metadata_payload() -> dict:
    return {
        "status": "OK",
        "data": {
            "latestVersion": {
                "files": [
                    {
                        "label": "File Manifest - Smart Meter Electricity Trial Data v1.0.1.pdf",
                        "restricted": False,
                        "directoryLabel": "Documentation",
                        "dataFile": {
                            "id": 793,
                            "filename": "File Manifest - Smart Meter Electricity Trial Data v1.0.1.pdf",
                            "filesize": 141005,
                        },
                    },
                    {
                        "label": "File1.txt.zip",
                        "restricted": True,
                        "directoryLabel": "Data",
                        "dataFile": {
                            "id": 806,
                            "filename": "File1.txt.zip",
                            "filesize": 101978611,
                        },
                    },
                ]
            }
        },
    }


@respx.mock
def test_download_cer_dataset_without_api_key_downloads_public_docs_only(tmp_path):
    respx.get("https://issda.ucd.ie/api/datasets/:persistentId/").mock(
        return_value=httpx.Response(200, json=_metadata_payload())
    )
    respx.get("https://issda.ucd.ie/api/access/datafile/793").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4")
    )

    result = download_cer_dataset(tmp_path)

    assert result["downloaded_public_files"] == ["File Manifest - Smart Meter Electricity Trial Data v1.0.1.pdf"]
    assert result["downloaded_restricted_files"] == []
    assert result["skipped_restricted_files"] == ["File1.txt.zip"]
    assert (tmp_path / "documentation" / "File Manifest - Smart Meter Electricity Trial Data v1.0.1.pdf").exists()
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["latestVersion"]["files"][0]["label"].startswith("File Manifest")


@respx.mock
def test_download_cer_dataset_with_api_key_downloads_restricted_files(tmp_path):
    respx.get("https://issda.ucd.ie/api/datasets/:persistentId/").mock(
        return_value=httpx.Response(200, json=_metadata_payload())
    )
    respx.get("https://issda.ucd.ie/api/access/datafile/793").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4")
    )
    respx.get("https://issda.ucd.ie/api/access/datafile/806").mock(
        return_value=httpx.Response(200, content=b"PK\x03\x04")
    )

    result = download_cer_dataset(tmp_path, api_key="secret-key")

    assert result["downloaded_restricted_files"] == ["File1.txt.zip"]
    assert result["skipped_restricted_files"] == []
    assert (tmp_path / "data" / "File1.txt.zip").exists()
