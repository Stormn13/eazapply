#!/usr/bin/env python3

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from openai import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError, InternalServerError, RateLimitError
from pydantic import BaseModel, Field, ValidationError

from extras.models import LLM7_MODEL, create_ingestion_model

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = ROOT / "user_data" / "career_sources"
DEFAULT_OKF_DIR = ROOT / "knowledge" / "okf"
DEFAULT_MANIFEST_PATH = ROOT / "knowledge" / "ingestion_manifest.json"
SOURCE_DIR = DEFAULT_SOURCE_DIR
OKF_DIR = DEFAULT_OKF_DIR
MANIFEST_PATH = DEFAULT_MANIFEST_PATH
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
RECORD_TYPE_DIRS = {
    "profile": OKF_DIR,
    "education": OKF_DIR,
    "skill": OKF_DIR,
    "project": OKF_DIR / "projects",
    "experience": OKF_DIR / "experience",
    "achievement": OKF_DIR / "achievements",
    "certification": OKF_DIR / "certifications",
    "activity": OKF_DIR / "activities",
}
DEFAULT_LLM_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_TIMEOUT_RETRIES = 2
MAX_LLM_TIMEOUT_RETRIES = 5


class SourceEvidence(BaseModel):
    source_file: str
    source_hash: str
    relevant_section: str | None = None
    extracted_claim: str


class CareerRecord(BaseModel):
    record_type: Literal[
        "profile",
        "education",
        "skill",
        "project",
        "experience",
        "achievement",
        "certification",
        "activity",
    ]
    record_id: str
    title: str
    status: Literal["draft", "active", "needs_verification"] = "draft"
    fields: dict[str, Any] = Field(default_factory=dict)
    evidence: list[SourceEvidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    normalization_original_values: dict[str, list[str]] = Field(default_factory=dict)


class CareerExtractionBundle(BaseModel):
    records: list[CareerRecord] = Field(default_factory=list)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def normalize_category_token(value: str) -> str:
    return re.sub(r"[\s-]+", "_", value.strip().casefold())


def position_context_text(record: dict[str, Any]) -> str:
    values: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            for nested in value.values():
                collect(nested)
        elif isinstance(value, list):
            for nested in value:
                collect(nested)

    collect(record.get("title", ""))
    collect(record.get("fields", {}))
    collect(record.get("evidence", []))
    return " ".join(values).casefold()


def normalize_position_type(record: dict[str, Any]) -> str:
    context = position_context_text(record)
    leadership_terms = (
        r"\bhead\b", r"\blead\b", r"\bleader\b", r"\bleadership\b",
        r"\bpresident\b", r"\bchair(?:person|man|woman)?\b", r"\bcaptain\b",
        r"\bofficer\b", r"\bsecretary\b", r"\btreasurer\b", r"\bdirector\b",
        r"\bmanager\b", r"\bfounder\b", r"\bcoordinator\b", r"\bresponsibility\b",
    )
    if any(re.search(term, context) for term in leadership_terms):
        return "experience"

    extracurricular_terms = (
        r"\bvolunteer(?:ing)?\b", r"\bextracurricular\b", r"\bunpaid\b",
        r"\bclub member\b", r"\bmember of (?:a )?(?:student )?club\b",
        r"\bparticipat(?:ed|ion|ing)\b", r"\bclub activity\b",
    )
    if any(re.search(term, context) for term in extracurricular_terms):
        return "activity"
    return "experience"


def normalize_extraction_payload(data: Any) -> Any:
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        return data

    canonical_record_types = {
        "profile", "education", "skill", "project", "experience",
        "achievement", "certification", "activity",
    }
    canonical_statuses = {"draft", "active", "needs_verification"}
    record_type_aliases = {
        "person": "profile",
        "competition": "achievement",
        "award": "achievement",
        "awards": "achievement",
        "leadership": "experience",
        "position_of_responsibility": "experience",
    }

    for index, record in enumerate(data["records"]):
        if not isinstance(record, dict):
            continue

        original_values: dict[str, str] = {}
        raw_record_type = record.get("record_type")
        if isinstance(raw_record_type, str):
            normalized_type = normalize_category_token(raw_record_type)
            record_type = record_type_aliases.get(normalized_type, normalized_type)
            if normalized_type == "position":
                record_type = normalize_position_type(record)
            if record_type not in canonical_record_types:
                raise ValueError(
                    f"Unsupported record_type {raw_record_type!r} in extracted record {index}; "
                    f"expected one of {', '.join(sorted(canonical_record_types))}."
                )
            if raw_record_type != record_type:
                original_values["record_type"] = raw_record_type
            record["record_type"] = record_type

        raw_status = record.get("status")
        if isinstance(raw_status, str):
            normalized_status = normalize_category_token(raw_status)
            status = "active" if normalized_status == "confirmed" else normalized_status
            if status not in canonical_statuses:
                raise ValueError(
                    f"Unsupported status {raw_status!r} in extracted record {index}; "
                    f"expected one of {', '.join(sorted(canonical_statuses))} or the known alias 'confirmed'."
                )
            if raw_status != status:
                original_values["status"] = raw_status
            record["status"] = status

        if original_values:
            stored_values = record.setdefault("normalization_original_values", {})
            if not isinstance(stored_values, dict):
                stored_values = {}
                record["normalization_original_values"] = stored_values
            for key, raw_value in original_values.items():
                prior = stored_values.get(key, [])
                if not isinstance(prior, list):
                    prior = [str(prior)]
                if raw_value not in prior:
                    prior.append(raw_value)
                stored_values[key] = prior

    return data


def ensure_directories() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OKF_DIR.mkdir(parents=True, exist_ok=True)
    for folder in RECORD_TYPE_DIRS.values():
        folder.mkdir(parents=True, exist_ok=True)


def configure_paths(source_dir: str | None, knowledge_dir: str | None) -> None:
    global SOURCE_DIR, OKF_DIR, MANIFEST_PATH, RECORD_TYPE_DIRS
    SOURCE_DIR = Path(source_dir).resolve() if source_dir else DEFAULT_SOURCE_DIR
    OKF_DIR = Path(knowledge_dir).resolve() if knowledge_dir else DEFAULT_OKF_DIR
    MANIFEST_PATH = OKF_DIR.parent / "ingestion_manifest.json"
    RECORD_TYPE_DIRS = {
        "profile": OKF_DIR,
        "education": OKF_DIR,
        "skill": OKF_DIR,
        "project": OKF_DIR / "projects",
        "experience": OKF_DIR / "experience",
        "achievement": OKF_DIR / "achievements",
        "certification": OKF_DIR / "certifications",
        "activity": OKF_DIR / "activities",
    }


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        return {"version": 1, "last_updated": None, "sources": []}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"version": 1, "last_updated": None, "sources": []}
        data.setdefault("version", 1)
        data.setdefault("last_updated", None)
        data.setdefault("sources", [])
        return data
    except json.JSONDecodeError:
        return {"version": 1, "last_updated": None, "sources": []}


def write_manifest(data: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supported_files() -> list[Path]:
    if not SOURCE_DIR.exists():
        return []
    files = []
    for path in sorted(SOURCE_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.name.lower() in {"readme.md", "README.md"}:
            continue
        if path.name.lower() == ".gitkeep":
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def extract_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            import pypdf
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("PDF support requires 'pypdf'. Install it or place a different supported file type.") from exc
        reader = pypdf.PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)

    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("DOCX support requires python-docx. Install it or place a different supported file type.") from exc
        document = Document(str(path))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    raise RuntimeError(f"Unsupported file type: {suffix}")


def parse_json_payload(raw_text: str) -> CareerExtractionBundle:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    data = json.loads(cleaned)
    data = normalize_extraction_payload(data)
    return CareerExtractionBundle.model_validate(data)


def llm_request_settings() -> tuple[float, int]:
    try:
        timeout_seconds = float(os.environ.get("LLM_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS)))
        retries = int(os.environ.get("LLM_TIMEOUT_RETRIES", str(DEFAULT_LLM_TIMEOUT_RETRIES)))
    except ValueError as exc:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a positive number and LLM_TIMEOUT_RETRIES must be an integer.") from exc

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a finite positive number.")
    if retries < 0 or retries > MAX_LLM_TIMEOUT_RETRIES:
        raise ValueError(f"LLM_TIMEOUT_RETRIES must be between 0 and {MAX_LLM_TIMEOUT_RETRIES}.")
    return timeout_seconds, retries


def is_retryable_llm_error(exc: Exception) -> bool:
    if isinstance(exc, AuthenticationError):
        return False
    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


def format_processing_error(exc: BaseException, timeout_seconds: float) -> str:
    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return f"LLM request timed out after {timeout_seconds:g} seconds: {exc}"
    return f"{type(exc).__name__}: {exc}"


def llm_extract_structured_data(source_path: Path, source_hash: str, text: str) -> CareerExtractionBundle:
    timeout_seconds, retries = llm_request_settings()
    model = create_ingestion_model(timeout_seconds)
    prompt = f"""
You are extracting structured career information from a source document.

Return valid JSON only.

Requirements:
- Use this exact schema:
  {{
    "records": [
      {{
        "record_type": "profile|education|skill|project|experience|achievement|certification|activity",
        "record_id": "stable-id-for-this-record",
        "title": "human readable title",
        "status": "draft",
        "fields": {{
          ...entity-specific details...
        }},
        "evidence": [
          {{
            "source_file": "filename",
            "source_hash": "sha256",
            "relevant_section": "section or page label if known",
            "extracted_claim": "exact claim or concise summary"
          }}
        ],
        "conflicts": []
      }}
    ]
  }}

Rules:
- Do not invent facts not present in the source.
- Keep the output conservative.
- If uncertain, set status to \"draft\" and include reasoning in conflicts.
- For skills, store them as individual skill records or grouped skills if appropriate.
- Preserve evidence for each record.
- Do not write markdown; only return JSON.

Source filename: {source_path.name}
Source hash: {source_hash}

Document content:
{text[:20000]}
"""
    for attempt in range(retries + 1):
        try:
            response = model.invoke(prompt)
            content = getattr(response, "content", str(response))
            return parse_json_payload(str(content))
        except Exception as exc:
            if attempt >= retries or not is_retryable_llm_error(exc):
                raise
            delay_seconds = min(2 ** attempt, 4)
            print(f"  Transient LLM error ({type(exc).__name__}); retrying in {delay_seconds} seconds ({attempt + 1}/{retries}).")
            time.sleep(delay_seconds)

    raise RuntimeError("LLM extraction ended without a result.")


def parse_existing_markdown_records() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"profile": [], "education": [], "skill": [], "project": [], "experience": [], "achievement": [], "certification": [], "activity": []}
    if not OKF_DIR.exists():
        return result

    for path in sorted(OKF_DIR.rglob("*.md")):
        if path.name == "README.md":
            continue
        if path.name == "index.md":
            continue
        if path.name == "profile.md":
            continue
        if path.name == "education.md":
            continue
        if path.name == "skills.md":
            continue

        text = path.read_text(encoding="utf-8", errors="replace")
        if not text.startswith("---\n"):
            continue
        parts = text.split("---\n", 2)
        if len(parts) < 3:
            continue
        frontmatter = parts[1]
        info: dict[str, Any] = {}
        for line in frontmatter.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()

        kind = info.get("kind")
        if kind in result:
            body_lines = parts[2].splitlines()
            fields: dict[str, Any] = {}
            evidence: list[SourceEvidence] = []
            conflicts: list[str] = []
            normalization_original_values: dict[str, list[str]] = {}
            section = ""
            evidence_item: dict[str, str] = {}

            def save_evidence_item() -> None:
                if {"source_file", "source_hash", "extracted_claim"}.issubset(evidence_item):
                    evidence.append(SourceEvidence(**evidence_item))
                evidence_item.clear()

            for line in body_lines:
                if line.startswith("## "):
                    if section == "Evidence":
                        save_evidence_item()
                    section = line[3:]
                    continue
                if section == "Structured fields" and line.startswith("- ") and ":" in line:
                    key, value = line[2:].split(":", 1)
                    try:
                        fields[key.strip()] = json.loads(value.strip())
                    except json.JSONDecodeError:
                        fields[key.strip()] = value.strip()
                elif section == "Evidence" and line.startswith("- "):
                    key, value = line[2:].split(":", 1)
                    evidence_key = {
                        "Source": "source_file",
                        "Source hash": "source_hash",
                        "Relevant section": "relevant_section",
                        "Extracted claim": "extracted_claim",
                    }.get(key.strip())
                    if evidence_key:
                        if evidence_key == "source_file" and "source_file" in evidence_item:
                            save_evidence_item()
                        evidence_item[evidence_key] = value.strip()
                elif section == "Original model values" and line.startswith("- ") and ":" in line:
                    key, value = line[2:].split(":", 1)
                    try:
                        normalization_original_values[key.strip()] = json.loads(value.strip())
                    except json.JSONDecodeError:
                        normalization_original_values[key.strip()] = [value.strip()]
                elif section == "Conflicts" and line.startswith("- "):
                    conflicts.append(line[2:].strip())
            if section == "Evidence":
                save_evidence_item()

            result[kind].append({
                "record_id": info.get("record_id", path.stem),
                "title": info.get("title", path.stem),
                "status": info.get("status", "draft"),
                "fields": fields,
                "evidence": evidence,
                "conflicts": conflicts,
                "normalization_original_values": normalization_original_values,
                "path": str(path),
            })
    return result


def find_existing_match(record: CareerRecord, existing: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    candidates = existing.get(record.record_type, [])
    for candidate in candidates:
        if candidate.get("record_id") == record.record_id:
            return candidate
        if normalize_name(str(candidate.get("title", ""))) == normalize_name(record.title):
            return candidate
    return None


def render_markdown(record: CareerRecord) -> str:
    frontmatter_lines = [
        "---",
        f"format: okf",
        f"version: 0.1",
        "namespace: personal-career-knowledge",
        f"kind: {record.record_type}",
        f"title: {record.title}",
        f"record_id: {record.record_id}",
        f"status: {record.status}",
        f"updated_at: {now_utc()}",
        "---",
        "",
        f"# {record.title}",
        "",
        "## Summary",
        "",
        "This record was generated by the automated career knowledge ingestion workflow.",
        "",
        "## Structured fields",
        "",
    ]
    for key, value in record.fields.items():
        frontmatter_lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")

    if record.evidence:
        frontmatter_lines.extend(["", "## Evidence", ""])
        for item in record.evidence:
            frontmatter_lines.append(f"- Source: {item.source_file}")
            frontmatter_lines.append(f"- Source hash: {item.source_hash}")
            if item.relevant_section:
                frontmatter_lines.append(f"- Relevant section: {item.relevant_section}")
            frontmatter_lines.append(f"- Extracted claim: {item.extracted_claim}")
            frontmatter_lines.append("")

    if record.conflicts:
        frontmatter_lines.extend(["", "## Conflicts", ""])
        for item in record.conflicts:
            frontmatter_lines.append(f"- {item}")

    if record.normalization_original_values:
        frontmatter_lines.extend(["", "## Original model values", ""])
        for key, values in record.normalization_original_values.items():
            frontmatter_lines.append(f"- {key}: {json.dumps(values, ensure_ascii=False)}")

    return "\n".join(frontmatter_lines).rstrip() + "\n"


def write_record(record: CareerRecord) -> tuple[bool, str]:
    folder = RECORD_TYPE_DIRS.get(record.record_type, OKF_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{record.record_id}.md"
    path = folder / filename
    path.write_text(render_markdown(record), encoding="utf-8")
    return True, str(path)


def merge_record(existing: CareerRecord | None, incoming: CareerRecord) -> CareerRecord:
    if existing is None:
        return incoming

    merged = existing.model_copy(deep=True)
    for key, value in incoming.fields.items():
        if key in merged.fields and merged.fields[key] != value:
            conflict = f"Field '{key}' has conflicting values across sources; the existing value was retained pending verification."
            if conflict not in merged.conflicts:
                merged.conflicts.append(conflict)
            merged.status = "needs_verification"
        else:
            merged.fields[key] = value
    known_evidence = {(item.source_file, item.source_hash, item.extracted_claim) for item in merged.evidence}
    for item in incoming.evidence:
        evidence_key = (item.source_file, item.source_hash, item.extracted_claim)
        if evidence_key not in known_evidence:
            merged.evidence.append(item)
            known_evidence.add(evidence_key)
    if incoming.conflicts:
        merged.conflicts.extend(item for item in incoming.conflicts if item not in merged.conflicts)
    for key, values in incoming.normalization_original_values.items():
        existing_values = merged.normalization_original_values.setdefault(key, [])
        merged.normalization_original_values[key] = list(dict.fromkeys(existing_values + values))
    if incoming.status == "needs_verification":
        merged.status = "needs_verification"
    return merged


def ingest_document(path: Path, dry_run: bool) -> tuple[dict[str, Any], list[CareerRecord]]:
    source_hash = file_hash(path)
    text = extract_text_from_file(path)
    bundle = llm_extract_structured_data(path, source_hash, text)

    existing = parse_existing_markdown_records()
    processed: list[CareerRecord] = []
    record_changes: list[dict[str, Any]] = []
    for record in bundle.records:
        match = find_existing_match(record, existing)
        if match:
            existing_record = CareerRecord(
                record_type=record.record_type,
                record_id=match["record_id"],
                title=match["title"],
                status=match["status"],
                fields=match.get("fields", {}),
                evidence=match.get("evidence", []),
                conflicts=match.get("conflicts", []),
                normalization_original_values=match.get("normalization_original_values", {}),
            )
            merged = merge_record(existing_record, record)
            processed.append(merged)
            record_changes.append({"record_id": merged.record_id, "is_new": False})
        else:
            processed.append(record)
            record_changes.append({"record_id": record.record_id, "is_new": True})

    if not dry_run:
        for record in processed:
            write_record(record)

    source_record = {
        "source_file": str(path.relative_to(ROOT)),
        "source_hash": source_hash,
        "ingested_at": now_utc(),
        "attempted_at": now_utc(),
        "status": "processed" if not dry_run else "dry_run",
        "record_ids": [record.record_id for record in processed],
        "errors": [],
    "record_changes": record_changes,
    }
    return source_record, processed


def summarize_changes(manifest: dict[str, Any], source_results: list[tuple[dict[str, Any], list[CareerRecord]]]) -> dict[str, Any]:
    summary = {
        "files_scanned": len(source_results),
        "new_files_processed": 0,
        "updated_files_processed": 0,
        "skipped_unchanged_files": 0,
        "failed_files": 0,
        "new_projects": 0,
        "updated_projects": 0,
        "new_skills": 0,
        "updated_skills": 0,
        "new_experience_records": 0,
        "updated_experience_records": 0,
        "conflicts_requiring_verification": 0,
    }

    for source_meta, records in source_results:
        if source_meta["status"] == "skipped":
            summary["skipped_unchanged_files"] += 1
            continue
        if source_meta["status"] == "failed":
            summary["failed_files"] += 1
            continue

        if source_meta["status"] not in {"processed", "dry_run"}:
            continue

        existing_entry = next((item for item in manifest.get("sources", []) if item.get("source_file") == source_meta["source_file"]), None)
        if existing_entry is None or existing_entry.get("status") != "processed":
            summary["new_files_processed"] += 1
        else:
            summary["updated_files_processed"] += 1

        record_changes = {item["record_id"]: item["is_new"] for item in source_meta.get("record_changes", [])}
        for record in records:
            is_new = record_changes.get(record.record_id, True)
            if record.record_type == "project":
                summary["new_projects" if is_new else "updated_projects"] += 1
            if record.record_type == "skill":
                summary["new_skills" if is_new else "updated_skills"] += 1
            if record.record_type == "experience":
                summary["new_experience_records" if is_new else "updated_experience_records"] += 1
            if record.status == "needs_verification":
                summary["conflicts_requiring_verification"] += 1
    return summary


PREVIEW_GROUPS = (
    ("Projects", "project"),
    ("Experience", "experience"),
    ("Skills", "skill"),
    ("Education", "education"),
    ("Achievements", "achievement"),
    ("Certifications", "certification"),
    ("Activities", "activity"),
    ("Profile", "profile"),
)
SKILL_NAME_ALIASES = {
    "ai_ml": "machine-learning",
    "ai_ml_and_data_science": "machine-learning",
    "cpp": "c-plus-plus",
    "c_plus_plus": "c-plus-plus",
    "js": "javascript",
    "nodejs": "node-js",
    "node_js": "node-js",
    "reactjs": "react",
    "react_js": "react",
    "ts": "typescript",
}


def record_field(record: CareerRecord, *names: str) -> Any:
    normalized_fields = {normalize_name(key).replace("-", "_"): value for key, value in record.fields.items()}
    for name in names:
        key = normalize_name(name).replace("-", "_")
        if key in normalized_fields:
            return normalized_fields[key]
    return None


def preview_record_title(record: CareerRecord) -> str:
    return record.title or str(record_field(record, "name", "skill", "role") or "Untitled record")


def preview_record_summary(record: CareerRecord) -> str:
    summary = record_field(record, "summary", "description", "overview", "details", "claim", "impact")
    if summary is not None:
        return json.dumps(summary, ensure_ascii=False) if not isinstance(summary, str) else summary
    for evidence in record.evidence:
        if evidence.extracted_claim:
            return evidence.extracted_claim
    return "Not provided by extraction."


def preview_record_dates(record: CareerRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.fields.items()
        if any(token in key.casefold() for token in ("date", "year", "duration", "start", "end"))
    }


def preview_records(source_results: list[tuple[dict[str, Any], list[CareerRecord]]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source_meta, records in source_results:
        source_file = Path(source_meta["source_file"]).name
        for record in records:
            entries.append({"record": record, "source_file": source_file})
    return entries


def print_record_preview(entries: list[dict[str, Any]]) -> None:
    print("\n=== Extracted Record Preview ===")
    for group_name, record_type in PREVIEW_GROUPS:
        grouped = [entry for entry in entries if entry["record"].record_type == record_type]
        print(f"\n{group_name} ({len(grouped)})")
        for index, entry in enumerate(grouped, 1):
            record: CareerRecord = entry["record"]
            print(f"  {index}. {preview_record_title(record)}")
            print(f"     record_type: {record.record_type}")
            print(f"     source: {entry['source_file']}")
            print(f"     status: {record.status}")
            print(f"     summary: {preview_record_summary(record)}")
            dates = preview_record_dates(record)
            if dates:
                print(f"     dates: {json.dumps(dates, ensure_ascii=False)}")
            if record.fields:
                print("     fields:")
                for key, value in record.fields.items():
                    print(f"       - {key}: {json.dumps(value, ensure_ascii=False)}")
            original_types = record.normalization_original_values.get("record_type", [])
            if original_types:
                print(f"     normalized_from: {', '.join(original_types)}")
            original_statuses = record.normalization_original_values.get("status", [])
            if original_statuses:
                print(f"     original_status: {', '.join(original_statuses)}")


def duplicate_entity_name(record: CareerRecord) -> str:
    if record.record_type == "skill":
        name = str(record_field(record, "name", "skill", "technology") or record.title)
        normalized = normalize_name(name).replace("-", "_")
        return SKILL_NAME_ALIASES.get(normalized, normalize_name(name))
    return normalize_name(record.title)


def duplicate_organization(record: CareerRecord) -> str:
    value = record_field(record, "organization", "company", "employer", "institution", "club")
    return normalize_name(str(value)) if value else ""


def records_may_be_same_entity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_record: CareerRecord = left["record"]
    right_record: CareerRecord = right["record"]
    if left_record.record_type != right_record.record_type or left["source_file"] == right["source_file"]:
        return False

    if left_record.record_type == "experience":
        left_org = duplicate_organization(left_record)
        right_org = duplicate_organization(right_record)
        if left_org and right_org and left_org != right_org:
            return False

    left_name = duplicate_entity_name(left_record)
    right_name = duplicate_entity_name(right_record)
    if left_name and left_name == right_name:
        return True
    if min(len(left_name), len(right_name)) >= 8 and SequenceMatcher(None, left_name, right_name).ratio() >= 0.9:
        return True
    return False


def duplicate_field_differences(entries: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    values_by_key: dict[str, list[str]] = {"title": []}
    for entry in entries:
        record: CareerRecord = entry["record"]
        values_by_key["title"].append(f"{entry['source_file']}: {record.title}")
        for key, value in record.fields.items():
            values_by_key.setdefault(key, []).append(f"{entry['source_file']}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}")

    differences = []
    for key, values in values_by_key.items():
        distinct = {value.split(": ", 1)[1] for value in values}
        if len(distinct) > 1:
            differences.append((key, values))
    return differences


def find_duplicate_groups(entries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    parents = list(range(len(entries)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index in range(len(entries)):
        for right_index in range(left_index + 1, len(entries)):
            if records_may_be_same_entity(entries[left_index], entries[right_index]):
                union(left_index, right_index)

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, entry in enumerate(entries):
        groups.setdefault(find(index), []).append(entry)
    return [
        group for group in groups.values()
        if len({entry["source_file"] for entry in group}) > 1
    ]


def print_duplicate_analysis(entries: list[dict[str, Any]]) -> None:
    groups = find_duplicate_groups(entries)
    print(f"\n=== Suspected Cross-Document Duplicates ({len(groups)}) ===")
    for index, group in enumerate(groups, 1):
        record_type = group[0]["record"].record_type
        titles = ", ".join(sorted({entry["record"].title for entry in group}))
        print(f"\n{index}. {record_type}: {titles}")
        print("   Why: same canonical type with identical or highly similar normalized entity names.")
        print("   Records:")
        for entry in group:
            record: CareerRecord = entry["record"]
            print(f"   - {entry['source_file']}: {record.title} [{record.record_id}]")
        differences = duplicate_field_differences(group)
        if differences:
            print("   Differing fields:")
            for key, values in differences:
                print(f"   - {key}: {'; '.join(values)}")
        else:
            print("   Differing fields: none among extracted titles/structured fields.")


def classification_warnings(entries: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    warnings: list[tuple[dict[str, Any], str]] = []
    role_pattern = r"\b(head|lead|leader|president|chair|captain|officer|intern|engineer|manager|director)\b"
    award_pattern = r"\b(award|competition|contest|hackathon|prize|winner|won|finalist|medal)\b"
    project_pattern = r"\b(project|built|developed|application|app|game|prototype|platform)\b"
    club_pattern = r"\b(club|society|student organization|student organisation)\b"

    for entry in entries:
        record: CareerRecord = entry["record"]
        context = " ".join([record.title, preview_record_summary(record), json.dumps(record.fields, ensure_ascii=False)]).casefold()
        title_context = record.title.casefold()
        if record.record_type == "experience" and re.search(award_pattern, context):
            warnings.append((entry, "Experience contains award/competition language; verify it belongs under achievements."))
        if record.record_type == "experience" and re.search(project_pattern, context) and not record_field(record, "organization", "company", "employer"):
            warnings.append((entry, "Experience contains project/build language without an employer field; verify it is not a project."))
        if record.record_type == "project" and re.search(role_pattern, title_context):
            warnings.append((entry, "Project title resembles a job or leadership role; verify it is not experience."))
        if record.record_type == "project" and any(normalize_name(key) in {"proficiency", "skill_level", "years_experience"} for key in record.fields):
            warnings.append((entry, "Project has skill/proficiency fields; verify it is not a skill record."))
        if record.record_type not in {"experience", "activity"} and re.search(club_pattern, context) and re.search(role_pattern, context):
            warnings.append((entry, "Club leadership/role signals conflict with this record type; review experience vs activity classification."))
    return warnings


def print_classification_warnings(entries: list[dict[str, Any]]) -> None:
    warnings = classification_warnings(entries)
    print(f"\n=== Suspicious Classifications ({len(warnings)}) ===")
    if not warnings:
        print("No deterministic classification warnings found.")
        return
    for entry, reason in warnings:
        record: CareerRecord = entry["record"]
        print(f"- {entry['source_file']}: {record.record_type} / {record.title}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest user career source documents into the OKF knowledge base.")
    parser.add_argument("--source-dir", default=None, help="Override the source directory containing career documents.")
    parser.add_argument("--knowledge-dir", default=None, help="Override the directory where OKF knowledge records and the manifest are stored.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and report what would change without updating the knowledge base.")
    parser.add_argument("--preview", action="store_true", help="Run read-only extraction and print grouped records, duplicate candidates, and classification warnings.")
    args = parser.parse_args()
    read_only = args.dry_run or args.preview

    configure_paths(args.source_dir, args.knowledge_dir)
    if not read_only:
        ensure_directories()

    manifest = load_manifest()
    original_manifest = json.loads(json.dumps(manifest))
    all_sources = supported_files()

    results: list[tuple[dict[str, Any], list[CareerRecord]]] = []
    for path in all_sources:
        print(f"Processing: {path.name}")
        print(f"Model: LLM7/{LLM7_MODEL}")
        current_hash = file_hash(path)
        previous = next((entry for entry in manifest.get("sources", []) if entry.get("source_file") == str(path.relative_to(ROOT))), None)

        if not args.preview and previous and previous.get("source_hash") == current_hash and previous.get("status") == "processed":
            print("Action: source unchanged; skipping extraction.")
            results.append((
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "source_hash": current_hash,
                    "ingested_at": previous.get("ingested_at"),
                    "status": "skipped",
                    "record_ids": previous.get("record_ids", []),
                    "errors": [],
                },
                [],
            ))
            continue

        print("Action: extracting career information...")
        try:
            source_info, records = ingest_document(path, read_only)
            results.append((source_info, records))
            print(f"✓ Extracted information from {path.name}")
        except (Exception, KeyboardInterrupt) as exc:  # pragma: no cover - behavior is intentionally resilient
            timeout_seconds = DEFAULT_LLM_TIMEOUT_SECONDS
            try:
                timeout_seconds, _ = llm_request_settings()
            except ValueError:
                pass
            error = format_processing_error(exc, timeout_seconds)
            attempted_at = now_utc()
            print(f"✗ Failed to process {path.name}")
            print(f"  Reason: {error}")
            print("  Status: failed")
            print("  The document can be retried on the next run.")
            results.append((
                {
                    "source_file": str(path.relative_to(ROOT)),
                    "source_hash": current_hash,
                    "ingested_at": attempted_at,
                    "attempted_at": attempted_at,
                    "status": "failed",
                    "record_ids": [],
                    "error": error,
                    "errors": [error],
                },
                [],
            ))

    if args.preview:
        entries = preview_records(results)
        print_record_preview(entries)
        print_duplicate_analysis(entries)
        print_classification_warnings(entries)

    manifest = json.loads(json.dumps(original_manifest))
    if not read_only:
        manifest["last_updated"] = now_utc()
        updated_sources = {entry.get("source_file"): entry for entry in manifest.get("sources", [])}
        for source_meta, _ in results:
            if source_meta["status"] in {"dry_run", "skipped"}:
                continue
            updated_sources[source_meta["source_file"]] = {
                "source_file": source_meta["source_file"],
                "source_hash": source_meta["source_hash"],
                "ingested_at": source_meta["ingested_at"],
                "attempted_at": source_meta.get("attempted_at", source_meta["ingested_at"]),
                "status": source_meta["status"],
                "record_ids": source_meta.get("record_ids", []),
                "errors": source_meta.get("errors", []),
            }
            if source_meta.get("error"):
                updated_sources[source_meta["source_file"]]["error"] = source_meta["error"]
        manifest["sources"] = list(updated_sources.values())
    if not read_only:
        write_manifest(manifest)

    summary = summarize_changes(original_manifest, results)
    print("Knowledge ingestion complete.")
    print(f"Files scanned: {summary['files_scanned']}")
    print(f"New files processed: {summary['new_files_processed']}")
    print(f"Updated files processed: {summary['updated_files_processed']}")
    print(f"Skipped unchanged files: {summary.get('skipped_unchanged_files', 0)}")
    print(f"Failed files: {summary['failed_files']}")
    print("Knowledge:")
    print(f"- New projects: {summary['new_projects']}")
    print(f"- Updated projects: {summary['updated_projects']}")
    print(f"- New experience records: {summary['new_experience_records']}")
    print(f"- Updated experience records: {summary['updated_experience_records']}")
    print(f"- New skills: {summary['new_skills']}")
    print(f"- Updated skills: {summary['updated_skills']}")
    print(f"- Conflicts requiring verification: {summary['conflicts_requiring_verification']}")
    if read_only:
        message = "preview" if args.preview else "dry run"
        print(f"No files were modified because this was a {message}.")


if __name__ == "__main__":
    main()
