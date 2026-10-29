import re
from pathlib import Path

from extras.models import llm7_model
from extras.states import ResumeState
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


ROOT_PATH = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT_PATH / "resume_format" / "format.tex"
OUTPUT_PATH = ROOT_PATH / "user_data" / "generated"
BODY_MARKER = r"\begin{document}"
FILENAME_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _extract_document_parts(template: str) -> tuple[str, str]:
    if template.count(BODY_MARKER) != 1 or template.count(r"\end{document}") != 1:
        raise ValueError(f"Resume template must contain one document start and end: {TEMPLATE_PATH}")
    preamble, _ = template.split(BODY_MARKER, 1)
    if r"\end{document}" not in template:
        raise ValueError("Resume template is missing \\end{document}.")
    return preamble.rstrip(), r"\end{document}"


def _clean_latex_body(content: str) -> str:
    body = content.strip()
    if body.startswith("```"):
        body = re.sub(r"^```(?:latex|tex)?\s*", "", body, flags=re.IGNORECASE)
        body = re.sub(r"\s*```$", "", body)
    if r"\begin{document}" in body:
        body = body.split(r"\begin{document}", 1)[1]
    if r"\end{document}" in body:
        body = body.split(r"\end{document}", 1)[0]
    body = body.strip()
    if not body:
        raise ValueError("LaTeX generator returned an empty document body.")
    if r"\documentclass" in body:
        raise ValueError("LaTeX generator must return document content, not a second preamble.")
    return body


def _generate_filename_stem(job_description: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Give the job description a concise filename label using exactly 2 or 3 words. "
            "Use the role or job title, not a company name. Return only the words, with no "
            "quotes, punctuation, or explanation."
        ),
        ("human", "Job description:\n{job_description}"),
    ])
    label = (prompt | llm7_model | StrOutputParser()).invoke({
        "job_description": job_description,
    })
    words = FILENAME_WORD_PATTERN.findall(label)
    if not 2 <= len(words) <= 3:
        raise ValueError(
            "LLM7 must return a 2- or 3-word job title for the resume filename; "
            f"received {label!r}."
        )
    stem = "-".join(word.lower() for word in words)
    if stem.upper() in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(COM|LPT)[1-9]", stem.upper()):
        stem = f"resume-{stem}"
    return stem


def _write_unique_latex(stem: str, latex: str) -> Path:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    suffix = 1
    while True:
        candidate_stem = stem if suffix == 1 else f"{stem}-{suffix}"
        existing_stems = {
            path.stem.casefold()
            for path in OUTPUT_PATH.iterdir()
            if path.is_file()
        }
        if candidate_stem.casefold() in existing_stems:
            suffix += 1
            continue

        latex_path = OUTPUT_PATH / f"{candidate_stem}.tex"
        try:
            with latex_path.open("x", encoding="utf-8") as latex_file:
                latex_file.write(latex)
        except FileExistsError:
            suffix += 1
            continue
        return latex_path


def resume_to_latex(state: ResumeState) -> dict[str, str]:
    resume = state.get("best_resume_text") or state.get("resume_text")
    job_description = state.get("jd")
    if not resume or not job_description:
        raise ValueError("LaTeX generation requires both a scored resume and job description.")
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Resume format template not found: {TEMPLATE_PATH}")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    preamble, document_end = _extract_document_parts(template)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create only the body of a LaTeX resume document. The supplied template is the "
            "format reference: reuse its commands and visual structure such as "
            "\\documentTitle, \\section, \\headingBf, \\headingIt, and resume_list. Do not output "
            "a preamble or document delimiters. Use only facts in the approved resume text; "
            "ignore personal details and achievements in the template's example body unless "
            "they also appear in the approved resume. Never invent or add placeholders. "
            "Return only valid LaTeX body content."
        ),
        (
            "human",
            "Format template:\n{template}\n\nApproved resume content:\n{resume}",
        ),
    ])
    body = (prompt | llm7_model | StrOutputParser()).invoke({
        "template": template,
        "resume": resume,
    })
    latex = f"{preamble}\n\n{BODY_MARKER}\n{_clean_latex_body(body)}\n\n{document_end}\n"

    filename_stem = _generate_filename_stem(job_description)
    latex_path = _write_unique_latex(filename_stem, latex)
    return {"latex_path": str(latex_path)}
