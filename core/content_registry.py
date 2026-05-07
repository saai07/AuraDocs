"""
Content Registry API client for AuraDocs.

Fetches pre-extracted educational content (OCR + LaTeX) from the
Content Registry microservice and feeds it into the local RAG pipeline.
"""

import json
import httpx
import streamlit as st
from core.config import CONTENT_REGISTRY_URL, CONTENT_REGISTRY_TIMEOUT


# ── HTTP helpers ──────────────────────────────────────────────────────

def _get(path: str, params: dict = None) -> dict | list | None:
    """Fire a GET request to the Content Registry API. Returns JSON or None on error."""
    url = f"{CONTENT_REGISTRY_URL}{path}"
    try:
        with httpx.Client(timeout=CONTENT_REGISTRY_TIMEOUT) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"Registry API error: {exc.response.status_code}")
    except httpx.ConnectError:
        st.error(
            "Cannot reach the Content Registry. "
            "Check that the server is running."
        )
    except httpx.TimeoutException:
        st.error("Content Registry request timed out. Try again in a moment.")
    except Exception as exc:
        st.error(f"Content Registry error: {exc}")
    return None


# ── Metadata lookups (cached) ────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_classes() -> list[str]:
    """Return a list of available class names from the registry."""
    data = _get("/api/metadata/classes")
    if data and isinstance(data, list):
        return data
    items = _get("/api/content")
    if items and isinstance(items, dict) and "items" in items:
        return sorted({i["class_name"] for i in items["items"] if i.get("class_name")})
    return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_subjects(class_name: str = None) -> list[str]:
    """Return available subjects, optionally filtered by class."""
    params = {"class_name": class_name} if class_name else None
    data = _get("/api/metadata/subjects", params=params)
    if data and isinstance(data, list):
        return data
    items = _get("/api/content", params=params)
    if items and isinstance(items, dict) and "items" in items:
        return sorted({i["subject"] for i in items["items"] if i.get("subject")})
    return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_chapters(class_name: str = None, subject: str = None) -> list[str]:
    """Return available chapters, optionally filtered."""
    params = {}
    if class_name:
        params["class_name"] = class_name
    if subject:
        params["subject"] = subject
    data = _get("/api/metadata/chapters", params=params)
    if data and isinstance(data, list):
        return data
    items = _get("/api/content", params=params)
    if items and isinstance(items, dict) and "items" in items:
        return sorted({i["chapter"] for i in items["items"] if i.get("chapter")})
    return []


# ── Content fetching ──────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def fetch_content_items(
    class_name: str = None,
    subject: str = None,
    chapter: str = None,
) -> list[dict]:
    """Fetch content items from the registry, filtered by metadata."""
    params = {}
    if class_name:
        params["class_name"] = class_name
    if subject:
        params["subject"] = subject
    if chapter:
        params["chapter"] = chapter
    data = _get("/api/content", params=params)
    if data and isinstance(data, dict) and "items" in data:
        return data["items"]
    if data and isinstance(data, list):
        return data
    return []


def build_registry_context(items: list[dict]) -> list[tuple[str, str]]:
    """Convert registry items into (text, source) tuples for the RAG pipeline."""
    results = []
    for item in items:
        text = item.get("extracted_text") or ""
        if not text.strip():
            continue
        parts = [
            item.get("class_name", ""),
            item.get("subject", ""),
            item.get("chapter", ""),
        ]
        source = " › ".join(p for p in parts if p) or item.get("filename", "registry")
        results.append((text, source))
    return results


def fetch_item_by_id(content_id: str) -> dict | None:
    """Fetch a single content item by ID via GET /api/content/{id}."""
    data = _get(f"/api/content/{content_id}")
    if data and isinstance(data, dict) and data.get("id"):
        return data
    # Fallback: find in listing
    all_data = _get("/api/content")
    items = []
    if all_data and isinstance(all_data, dict) and "items" in all_data:
        items = all_data["items"]
    elif all_data and isinstance(all_data, list):
        items = all_data
    for item in items:
        if item.get("id") == content_id:
            return item
    return None


def fetch_extraction_progress(content_id: str) -> dict | None:
    """Poll GET /api/content/{id}/progress for extraction status."""
    return _get(f"/api/content/{content_id}/progress")


# ── Upload + Extract (SSE streaming) ─────────────────────────────────

def upload_and_extract_stream(
    file_bytes: bytes,
    filename: str,
    class_name: str,
    subject: str,
    chapter: str,
    tags: str = "[]",
):
    """
    Upload a file and extract text via the SSE /upload-extract endpoint.

    Yields progress dicts from the server's Server-Sent Events stream:
        {"percent": 42, "message": "Extracting page 3/7...", "status": "processing"}
        {"percent": 100, ..., "status": "completed", "content_id": "...", "extracted_text": "..."}
    """
    url = f"{CONTENT_REGISTRY_URL}/api/content/upload-extract"
    # Long read timeout — extraction can take 30s+ per page on CPU
    timeout = httpx.Timeout(connect=30, read=300, write=30, pool=30)

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                url,
                data={
                    "class_name": class_name,
                    "subject": subject,
                    "chapter": chapter,
                    "description": "",
                    "tags": tags,
                },
                files={"file": (filename, file_bytes)},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            yield event
                            if event.get("status") in ("completed", "failed"):
                                return
                        except json.JSONDecodeError:
                            continue
                    # Ignore keepalive comments (": keepalive") and empty lines
    except httpx.HTTPStatusError as exc:
        yield {"percent": 0, "message": f"Server error: {exc.response.status_code}", "status": "failed"}
    except httpx.ConnectError:
        yield {"percent": 0, "message": "Cannot reach the Content Registry server.", "status": "failed"}
    except httpx.TimeoutException:
        yield {"percent": 0, "message": "Request timed out.", "status": "failed"}
    except Exception as exc:
        yield {"percent": 0, "message": f"Error: {exc}", "status": "failed"}
