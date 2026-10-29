import shutil
import subprocess
from pathlib import Path

from extras.states import ResumeState


def latex_to_pdf(state: ResumeState) -> dict[str, str]:
    latex_path_value = state.get("latex_path")
    if not latex_path_value:
        raise ValueError("PDF conversion requires latex_path.")

    latex_path = Path(latex_path_value).resolve()
    if not latex_path.is_file():
        raise FileNotFoundError(f"Generated LaTeX file not found: {latex_path}")

    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise RuntimeError(
            "pdflatex is required to create the resume PDF. Install a LaTeX distribution "
            "such as MiKTeX or TeX Live and ensure pdflatex is on PATH."
        )

    try:
        result = subprocess.run(
            [
                compiler,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                "-output-directory",
                str(latex_path.parent),
                str(latex_path),
            ],
            cwd=latex_path.parent,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("LaTeX compilation timed out after 120 seconds.") from exc

    pdf_path = latex_path.with_suffix(".pdf")
    if result.returncode != 0 or not pdf_path.is_file():
        compiler_output = "\n".join((result.stdout + "\n" + result.stderr).splitlines()[-30:])
        raise RuntimeError(
            f"LaTeX compilation failed with exit code {result.returncode}.\n{compiler_output}"
        )

    return {"pdf_path": str(pdf_path)}
