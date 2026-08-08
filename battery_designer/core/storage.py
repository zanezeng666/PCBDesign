from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

from .errors import DesignError


class ProjectStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, payload: dict) -> tuple[str, Path]:
        project_id = uuid4().hex
        directory = self.root / project_id
        directory.mkdir(parents=True)
        self.write_json(project_id, "project.json", {"id": project_id, "status": "created", **payload})
        return project_id, directory

    def directory(self, project_id: str) -> Path:
        if not re.fullmatch(r"[a-f0-9]{32}", project_id):
            raise DesignError("INVALID_PROJECT_ID", "Invalid project id.", status_code=404)
        directory = (self.root / project_id).resolve()
        if directory.parent != self.root or not directory.exists():
            raise DesignError("PROJECT_NOT_FOUND", "Project not found.", status_code=404)
        return directory

    def read(self, project_id: str) -> dict:
        return json.loads((self.directory(project_id) / "project.json").read_text(encoding="utf-8"))

    def update(self, project_id: str, **updates) -> dict:
        project = self.read(project_id)
        project.update(updates)
        self.write_json(project_id, "project.json", project)
        return project

    def write_json(self, project_id: str, name: str, data: dict) -> Path:
        directory = self.root / project_id if re.fullmatch(r"[a-f0-9]{32}", project_id) else self.directory(project_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def artifact(self, project_id: str, relative_path: str) -> Path:
        root = self.directory(project_id)
        path = (root / relative_path).resolve()
        if root not in path.parents or not path.is_file():
            raise DesignError("ARTIFACT_NOT_FOUND", "Artifact not found.", status_code=404)
        return path
