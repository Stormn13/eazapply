# OKF Career Knowledge Base

This directory is the canonical structured knowledge base for the user's career data.

## Purpose

- Store personal career facts in a structured, retrievable format
- Keep source-backed evidence linked to upstream documents
- Support incremental updates without overwriting prior knowledge
- Serve as the long-term source for future resume generation and tailoring

## Workflow

1. Add career documents to `user_data/career_sources/`
2. Run `python populate_knowledge.py`
3. The ingestion script extracts relevant information from supported documents
4. Structured records are created or updated in this OKF folder
5. Re-running ingestion later will process new or changed source files only

## Important rules

- Source documents are inputs.
- `knowledge/okf/` is the canonical knowledge base.
- Resume generation is downstream and must not write directly to the source documents.
- This directory is intentionally schema-first and template-driven until real user data is populated.

## Supported categories

- Profile
- Education
- Skills
- Projects
- Experience
- Achievements
- Certifications
- Activities
