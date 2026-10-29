import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from extras.models import llm7_model
from extras.states import ResumeState
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


MAX_REPAIR_ATTEMPTS = 2


def _has_balanced_braces(source: str) -> bool:
    depth = 0
    escaped = False
    in_comment = False

    for character in source:
        if character == "\n":
            in_comment = False
            escaped = False
            continue
        if in_comment:
            continue
        if character == "%" and not escaped:
            in_comment = True
            continue
        if character == "{" and not escaped:
            depth += 1
        elif character == "}" and not escaped:
            depth -= 1
            if depth < 0:
                return False
        if character == "\\":
            escaped = not escaped
        else:
            escaped = False

    return depth == 0


def _clean_repaired_latex(response: str) -> str:
    source = response.strip()
    if source.startswith("```"):
        source = re.sub(r"^```(?:latex|tex)?\s*", "", source, flags=re.IGNORECASE)
        source = re.sub(r"\s*```$", "", source)
    if not source:
        raise ValueError("The LaTeX repair model returned an empty document.")
    if source.count(r"\begin{document}") != 1 or source.count(r"\end{document}") != 1:
        raise ValueError("The LaTeX repair model did not return one complete document.")
    if not _has_balanced_braces(source):
        raise ValueError("The LaTeX repair model returned unbalanced braces.")
    return source


def _repair_latex(source: str, diagnostics: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Repair LaTeX compilation errors in the supplied complete resume source. Return the "
            "entire corrected LaTeX document only, not a code fence or explanation. Make the "
            "smallest syntax changes needed to compile. Preserve all resume facts, formatting, "
            "and template commands; do not invent, delete, or rewrite resume content."
        ),
        (
            "human",
            "Compiler diagnostics:\n{diagnostics}\n\nComplete LaTeX source:\n{source}",
        ),
    ])
    response = (prompt | llm7_model | StrOutputParser()).invoke({
        "diagnostics": diagnostics,
        "source": source,
    })
    return _clean_repaired_latex(response)


def _compile_in_temporary_directory(
    compiler: str,
    latex_path: Path,
    source: str,
) -> tuple[subprocess.CompletedProcess[str], bytes | None, str]:
    with tempfile.TemporaryDirectory(
        prefix="resume-latex-",
        dir=latex_path.parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        staged_tex = temporary_path / latex_path.name
        staged_tex.write_text(source, encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    compiler,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-file-line-error",
                    "-output-directory",
                    str(temporary_path),
                    str(staged_tex),
                ],
                cwd=temporary_path,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("LaTeX compilation timed out after 120 seconds.") from exc

        staged_pdf = staged_tex.with_suffix(".pdf")
        pdf_content = staged_pdf.read_bytes() if staged_pdf.is_file() else None
        staged_log = staged_tex.with_suffix(".log")
        log_content = staged_log.read_text(encoding="utf-8", errors="replace") if staged_log.is_file() else ""
        return result, pdf_content, log_content


def latex_to_pdf(state: ResumeState) -> dict[str, str]:
    latex_path_value = state.get("latex_path")
    if not latex_path_value:
        raise ValueError("PDF conversion requires latex_path.")

    latex_path = Path(latex_path_value).resolve()
    if not latex_path.is_file():
        raise FileNotFoundError(f"Generated LaTeX file not found: {latex_path}")
    current_source = latex_path.read_text(encoding="utf-8")

    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError(
            "pdflatex is required to create the resume PDF. Install a LaTeX distribution "
            "such as MiKTeX or TeX Live and ensure pdflatex is on PATH."
        )

    pdf_path = latex_path.with_suffix(".pdf")
    for repair_attempt in range(MAX_REPAIR_ATTEMPTS + 1):
        result, pdf_content, log_content = _compile_in_temporary_directory(
            compiler,
            latex_path,
            current_source,
        )
        if result.returncode == 0 and pdf_content is not None:
            latex_path.write_text(current_source, encoding="utf-8")
            pdf_path.write_bytes(pdf_content)
            if log_content:
                latex_path.with_suffix(".log").write_text(log_content, encoding="utf-8")
            return {"pdf_path": str(pdf_path)}

        diagnostics = "\n".join(
            (
                f"pdflatex exited with status {result.returncode}.",
                result.stdout,
                result.stderr,
                "\n".join(log_content.splitlines()[-80:]),
            )
        ).strip()
        if log_content:
            latex_path.with_suffix(".log").write_text(log_content, encoding="utf-8")
        if repair_attempt == MAX_REPAIR_ATTEMPTS:
            latex_path.write_text(current_source, encoding="utf-8")
            raise RuntimeError(
                f"LaTeX compilation failed after {MAX_REPAIR_ATTEMPTS} repair attempts. "
                f"The failed source and compiler log were retained at {latex_path}.\n"
                f"{diagnostics}"
            )

        try:
            current_source = _repair_latex(current_source, diagnostics)
        except ValueError as exc:
            diagnostics = f"{diagnostics}\nLaTeX repair response was rejected: {exc}"
            if repair_attempt + 1 == MAX_REPAIR_ATTEMPTS:
                latex_path.write_text(current_source, encoding="utf-8")
                raise RuntimeError(
                    f"LaTeX compilation failed after {repair_attempt + 1} repair attempts. "
                    f"The failed source and compiler log were retained at {latex_path}.\n"
                    f"{diagnostics}"
                ) from exc

    raise RuntimeError("LaTeX compilation ended without producing a PDF.")
