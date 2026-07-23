from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml

from .errors import DesignError


def normalize_mpn(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


@dataclass(frozen=True)
class DevicePackage:
    requested_model: str
    full_mpn: str
    manufacturer: str
    package: str
    supported_series: tuple[int, ...]
    port_topologies: tuple[str, ...]
    status: str
    source_url: str
    source_sha256: str
    pins: dict[str, str]
    parameters: dict[str, Any]
    marking: dict[str, Any]
    reference_components: dict[str, Any]
    source_cache: str | None
    template_dir: str | None
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], requested_model: str) -> "DevicePackage":
        required = ["full_mpn", "manufacturer", "package", "supported_series", "port_topologies", "source_url", "pins"]
        missing = [field for field in required if not data.get(field)]
        if missing:
            raise DesignError("IC_PACKAGE_INCOMPLETE", "IC package metadata is incomplete.", {"missing": missing})
        canonical_data = {key: value for key, value in data.items() if not key.startswith("_")}
        canonical = json.dumps(canonical_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return cls(
            requested_model=requested_model,
            full_mpn=data["full_mpn"],
            manufacturer=data["manufacturer"],
            package=data["package"],
            supported_series=tuple(int(v) for v in data["supported_series"]),
            port_topologies=tuple(data["port_topologies"]),
            status=data.get("status", "candidate"),
            source_url=data["source_url"],
            source_sha256=data.get("source_sha256") or hashlib.sha256(canonical).hexdigest(),
            pins={str(k): str(v) for k, v in data["pins"].items()},
            parameters=dict(data.get("parameters", {})),
            marking=dict(data.get("marking", {})),
            reference_components=dict(data.get("reference_components", {})),
            source_cache=data.get("source_cache"),
            template_dir=data.get("template_dir"),
            confidence=float(data.get("confidence", 0.0)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "full_mpn": self.full_mpn,
            "manufacturer": self.manufacturer,
            "package": self.package,
            "supported_series": list(self.supported_series),
            "port_topologies": list(self.port_topologies),
            "status": self.status,
            "source_url": self.source_url,
            "source_sha256": self.source_sha256,
            "pins": self.pins,
            "parameters": self.parameters,
            "marking": self.marking,
            "reference_components": self.reference_components,
            "source_cache": self.source_cache,
            "template_dir": self.template_dir,
            "confidence": self.confidence,
        }


class IcCatalog:
    """Versioned local catalog with an optional structured online resolver.

    The resolver endpoint must accept GET ?q=<model> and return either a list
    or {"candidates": [...]}. Search-result HTML is deliberately not treated as
    trusted electrical metadata.
    """

    def __init__(self, catalog_dir: Path, cache_dir: Path, resolver_endpoint: str | None = None):
        self.catalog_dir = catalog_dir
        self.cache_dir = cache_dir
        self.resolver_endpoint = resolver_endpoint or os.getenv("IC_RESOLVER_ENDPOINT")

    def _local_candidates(self) -> list[dict[str, Any]]:
        candidates = []
        if not self.catalog_dir.exists():
            return candidates
        for path in sorted(self.catalog_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["_catalog_path"] = str(path)
            candidates.append(data)
        return candidates

    def resolve(self, requested_model: str) -> DevicePackage:
        requested = normalize_mpn(requested_model)
        local = [item for item in self._local_candidates() if normalize_mpn(item["full_mpn"]) == requested or requested in [normalize_mpn(v) for v in item.get("aliases", [])]]
        if local:
            return self._apply_validation(DevicePackage.from_dict(self._rank(local, requested)[0], requested_model))
        cached = self.cache_dir / f"{requested}.json"
        if cached.exists():
            payload = json.loads(cached.read_text(encoding="utf-8"))
            return self._apply_validation(DevicePackage.from_dict(self._rank(payload, requested)[0], requested_model))
        if not self.resolver_endpoint:
            raise DesignError(
                "IC_NOT_RESOLVED",
                "The IC is not in the local catalog and no IC_RESOLVER_ENDPOINT is configured.",
                {"requested_model": requested_model},
            )
        try:
            response = httpx.get(self.resolver_endpoint, params={"q": requested_model}, timeout=20.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise DesignError("IC_LOOKUP_FAILED", "The online IC resolver failed.", {"reason": str(exc)}) from exc
        candidates = payload.get("candidates", []) if isinstance(payload, dict) else payload
        if not candidates:
            raise DesignError("IC_NOT_RESOLVED", "No trustworthy IC candidate was returned.", {"requested_model": requested_model})
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._apply_validation(DevicePackage.from_dict(self._rank(candidates, requested)[0], requested_model))

    def promote(self, device: DevicePackage, validation: dict[str, Any]) -> DevicePackage:
        if not device.template_dir:
            raise DesignError("TEMPLATE_REQUIRED_FOR_VALIDATION", "A hardware-tested design cannot be promoted without a versioned KiCad template.")
        record = {
            "full_mpn": device.full_mpn,
            "source_sha256": device.source_sha256,
            "template_dir": device.template_dir,
            "validation": validation,
        }
        validation_dir = self.cache_dir / "validated"
        validation_dir.mkdir(parents=True, exist_ok=True)
        path = validation_dir / f"{normalize_mpn(device.full_mpn)}.json"
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return replace(device, status="validated")

    def _apply_validation(self, device: DevicePackage) -> DevicePackage:
        path = self.cache_dir / "validated" / f"{normalize_mpn(device.full_mpn)}.json"
        if not path.exists():
            return device
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("source_sha256") != device.source_sha256 or record.get("template_dir") != device.template_dir:
            return device
        return replace(device, status="validated")

    @staticmethod
    def _rank(candidates: list[dict[str, Any]], requested: str) -> list[dict[str, Any]]:
        def score(item: dict[str, Any]) -> tuple:
            exact = normalize_mpn(item.get("full_mpn", "")) == requested
            official = bool(item.get("official_source", False))
            reference = bool(item.get("reference_circuit_verified", False))
            confidence = float(item.get("confidence", 0.0))
            return (exact, official, reference, confidence, item.get("manufacturer", ""), item.get("full_mpn", ""))
        return sorted(candidates, key=score, reverse=True)


def validate_ic_for_design(device: DevicePackage, series_cells: int, topology: str) -> None:
    if series_cells not in device.supported_series:
        raise DesignError(
            "IC_SERIES_MISMATCH",
            f"{device.full_mpn} does not support {series_cells} series cells.",
            {"supported_series": list(device.supported_series)},
        )
    if topology not in device.port_topologies:
        raise DesignError(
            "IC_PORT_TOPOLOGY_MISMATCH",
            f"{device.full_mpn} does not have a reviewed {topology} port topology.",
            {"supported": list(device.port_topologies)},
        )


def get_reference_mosfet_mpn(device: DevicePackage) -> str:
    """Return the recommended MOSFET MPN for this IC.

    Looks up ``reference_components.M1.mpn`` (or ``M2.mpn``),
    falling back to the standard companion FS8205A.
    """
    ref = device.reference_components
    for key in ("M1", "M2"):
        entry = ref.get(key)
        if isinstance(entry, dict) and entry.get("mpn"):
            return entry["mpn"]
    return "FS8205A"
