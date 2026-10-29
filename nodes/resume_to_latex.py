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


def resume_to_latex(state: ResumeState) -> dict[str, str]:
    resume = state.get("best_resume_text") or state.get("resume_text")
    if not resume:
        raise ValueError("LaTeX generation requires a scored resume.")
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

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    latex_path = OUTPUT_PATH / "resume.tex"
    latex_path.write_text(latex, encoding="utf-8")
    return {"latex_path": str(latex_path)}
