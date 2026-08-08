from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DesignError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    status_code: int = 422

    def __str__(self) -> str:
        return self.message

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }
