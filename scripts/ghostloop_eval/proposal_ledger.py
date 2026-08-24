"""Append-only, hash-chained proposal audit ledger."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def proposal_id(
    run_id: str,
    source: str,
    round_id: int,
    source_id: int,
    target_id: int,
    region_id: str | None = None,
) -> str:
    material = {
        "run_id": run_id,
        "source": source,
        "round": int(round_id),
        "source_id": int(source_id),
        "target_id": int(target_id),
        "region_id": region_id,
    }
    return hashlib.sha256(_canonical(material)).hexdigest()[:20]


class ProposalLedger:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = str(run_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"Refusing to overwrite proposal ledger: {self.path}")
        self._previous_hash = GENESIS_HASH

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **record,
            "previous_hash": self._previous_hash,
        }
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        payload["record_hash"] = digest
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._previous_hash = digest
        return payload

    @staticmethod
    def verify(path: Path) -> list[dict[str, Any]]:
        previous = GENESIS_HASH
        records = []
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                digest = record.pop("record_hash")
                if record.get("previous_hash") != previous:
                    raise ValueError(f"Broken ledger chain at line {line_number}")
                expected = hashlib.sha256(_canonical(record)).hexdigest()
                if digest != expected:
                    raise ValueError(f"Invalid ledger hash at line {line_number}")
                record["record_hash"] = digest
                records.append(record)
                previous = digest
        return records
