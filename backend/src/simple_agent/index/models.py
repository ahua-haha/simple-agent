"""Index node database and in-memory tree models."""

from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field as PydanticField
from sqlalchemy import Column, String
from sqlmodel import Field, SQLModel


class IndexNodeRecord(SQLModel, table=True):
    __tablename__ = "index_nodes"

    path: str = Field(primary_key=True)
    kind: str = Field(default="file", index=True)
    metadata_json: str = Field(default="{}", sa_column=Column("metadata", String))
    status: str = Field(default="updated", index=True)
    propagation_count: int = Field(default=4)
    updated_at: int = Field(default_factory=lambda: int(time.time()))

    @property
    def type(self) -> str:
        node = index_node_from_record(self)
        if isinstance(node, SymbolNode):
            return node.symbol_type
        return node.kind

    @property
    def description(self) -> str:
        return index_node_from_record(self).description


class BaseNode(BaseModel):
    path: str
    kind: str
    name: str = ""
    description: str = ""
    propagation_count: int = 4
    updated_at: int = PydanticField(default_factory=lambda: int(time.time()))
    children: list[BaseNode] = PydanticField(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        if not self.name:
            self.name = Path(self.path).name or self.path

    @property
    def is_dir(self) -> bool:
        return self.kind == "directory"

    def metadata_json(self) -> str:
        return self.model_dump_json(
            exclude={"path", "kind", "propagation_count", "updated_at", "children"}
        )

    def comment(self) -> str:
        return self.description

    def format_node(self, label: str | None = None) -> str:
        node_label = label if label is not None else self.name
        suffix = "/" if self.is_dir else ""
        comment = self.comment()
        if comment:
            return f"{node_label}{suffix}  # {comment}"
        return f"{node_label}{suffix}"


class FileNode(BaseNode):
    kind: Literal["file"] = "file"

    @classmethod
    def from_record(cls, record: IndexNodeRecord) -> "FileNode":
        return cls(
            path=record.path,
            propagation_count=record.propagation_count,
            updated_at=record.updated_at,
            **_metadata_dict(record.metadata_json),
        )


class DirectoryNode(BaseNode):
    kind: Literal["directory"] = "directory"

    @classmethod
    def from_record(cls, record: IndexNodeRecord) -> "DirectoryNode":
        return cls(
            path=record.path,
            propagation_count=record.propagation_count,
            updated_at=record.updated_at,
            **_metadata_dict(record.metadata_json),
        )


class SymbolNode(BaseNode):
    kind: Literal["symbol"] = "symbol"
    symbol_type: str = "symbol"
    line_start: int | None = None
    line_end: int | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.name:
            self.name = self.path.rsplit(":", 1)[-1]

    def comment(self) -> str:
        parts = []
        if self.description:
            parts.append(self.description)
        if self.line_start is not None:
            if self.line_end is not None and self.line_end != self.line_start:
                parts.append(f"lines {self.line_start}-{self.line_end}")
            else:
                parts.append(f"line {self.line_start}")
        if self.symbol_type:
            parts.append(f"[{self.symbol_type}]")
        return " ".join(parts)

    @classmethod
    def from_record(cls, record: IndexNodeRecord) -> "SymbolNode":
        return cls(
            path=record.path,
            propagation_count=record.propagation_count,
            updated_at=record.updated_at,
            **_metadata_dict(record.metadata_json),
        )


IndexNode = DirectoryNode | FileNode | SymbolNode


def index_node_from_record(record: IndexNodeRecord) -> IndexNode:
    if record.kind == "symbol":
        return SymbolNode.from_record(record)
    if record.kind == "directory":
        return DirectoryNode.from_record(record)
    return FileNode.from_record(record)


def index_node_to_record(node: IndexNode) -> IndexNodeRecord:
    return IndexNodeRecord(
        path=node.path,
        kind=node.kind,
        metadata_json=node.metadata_json(),
        status="updated",
        propagation_count=node.propagation_count,
        updated_at=node.updated_at,
    )


def _metadata_dict(metadata: str) -> dict:
    payload = json.loads(metadata or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Index node metadata must be a JSON object")
    return payload
