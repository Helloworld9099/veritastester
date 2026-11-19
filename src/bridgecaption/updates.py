"""Non-blocking update discovery. Installation requires a trusted packaged updater.

Never download and execute arbitrary release assets from a JSON response.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


@dataclass(frozen=True)
class Update:
    version: str
    url: str


def current_version() -> str:
    try:
        return version("bridgecaption")
    except PackageNotFoundError:
        return "0.1.0"


def version_tuple(value: str) -> tuple[int, ...]:
    parts = value.strip().lstrip("v").split(".")
    if not parts or not all(part.isdecimal() for part in parts):
        raise ValueError("Invalid update version")
    return tuple(int(part) for part in parts)


def find_update(current: str | None = None) -> Update | None:
    """Look up a release. No token or private credentials are transmitted."""
    endpoint = os.getenv(
        "BRIDGECAPTION_UPDATE_ENDPOINT",
        "https://api.github.com/repos/Helloworld9099/veritastester/releases/latest",
    )
    if not endpoint.startswith("https://"):
        raise ValueError("Update endpoint must use HTTPS")
    request = Request(endpoint, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "BridgeCaption-Updater",
    })
    try:
        with urlopen(request, timeout=8) as response:
            if response.status != 200:
                raise RuntimeError("Update service unavailable")
            data = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("Could not check for updates") from exc
    if not isinstance(data, dict) or not isinstance(data.get("tag_name"), str):
        raise RuntimeError("Invalid update response")
    tag = data["tag_name"]
    if data.get("draft") or data.get("prerelease"):
        return None
    if version_tuple(tag) <= version_tuple(current or current_version()):
        return None
    url = data.get("html_url", "")
    if not isinstance(url, str) or not url.startswith("https://github.com/"):
        raise RuntimeError("Invalid update destination")
    return Update(version=tag.lstrip("v"), url=url)
