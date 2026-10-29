# Career Source Documents

Place your career-related documents in this folder before running the knowledge ingestion workflow.

## What belongs here

You can place any of the following here without renaming files:

- resumes
- CVs
- cover letters
- project writeups
- internship documents
- certificates
- portfolios
- other career-related PDFs, DOCX, TXT, or Markdown files

## How to use

1. Add one or more career documents to this directory.
2. Run the ingestion script from the project root.
3. The ingestion process will extract structured information and update the OKF knowledge base.

## Important

- This folder is an input source, not the canonical knowledge base.
- The system supports incremental updates.
- Running ingestion again will process new or changed files without deleting existing knowledge.

Example:

```bash
python populate_knowledge.py
python populate_knowledge.py --dry-run
```
