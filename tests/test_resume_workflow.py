import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import nodes.ats_score as ats_node
import nodes.jd_to_resume as draft_node
import nodes.latex_to_pdf as pdf_node
import nodes.resume_to_latex as latex_node


class AtsWorkflowTests(unittest.TestCase):
    def test_parser_validates_score_and_feedback(self) -> None:
        self.assertEqual(
            ats_node._parse_ats_result('{"score": 8.75, "feedback": ["Add supported keywords"]}'),
            (8.75, ["Add supported keywords"]),
        )
        with self.assertRaises(ValueError):
            ats_node._parse_ats_result('{"score": 11, "feedback": []}')
        with self.assertRaises(ValueError):
            ats_node._parse_ats_result('{"score": 8, "feedback": "not a list"}')

    def test_ats_node_tracks_scores_and_preserves_best_resume(self) -> None:
        model = RunnableLambda(
            lambda _: AIMessage(content=json.dumps({"score": 8.4, "feedback": ["Clarify project impact"]}))
        )
        with patch.object(ats_node, "llm7_model", model):
            result = ats_node.ats_score({
                "jd": "Python developer",
                "resume_text": "Current draft",
                "ats_score_history": [8.1],
                "best_resume_text": "Earlier draft",
                "best_ats_score": 8.2,
            })

        self.assertEqual(result["ats_score_history"], [8.1, 8.4])
        self.assertEqual(result["best_resume_text"], "Current draft")
        self.assertEqual(result["best_ats_score"], 8.4)

    def test_router_stops_on_target_plateau_regression_or_iteration_cap(self) -> None:
        cases = [
            ({"ats_score_history": [9.5], "resume_iteration": 1}, "generate_latex"),
            ({"ats_score_history": [8.0, 8.05], "resume_iteration": 2}, "generate_latex"),
            ({"ats_score_history": [8.0, 7.9], "resume_iteration": 2}, "generate_latex"),
            ({"ats_score_history": [8.0, 8.2], "resume_iteration": 5}, "generate_latex"),
            ({"ats_score_history": [8.0, 8.2], "resume_iteration": 2}, "revise_resume"),
        ]
        for state, expected in cases:
            with self.subTest(state=state):
                self.assertEqual(ats_node.route_after_ats_score(state), expected)

    def test_revision_uses_feedback_and_recursively_loaded_source_resumes(self) -> None:
        captured = {}

        def capture_prompt(prompt_value):
            captured["human"] = prompt_value.to_messages()[1].content
            return "Revised resume"

        with patch.object(draft_node, "llm7_model", RunnableLambda(capture_prompt)):
            result = draft_node.jd_to_resume({
                "jd": "Python developer",
                "resume_text": "Current draft",
                "ats_feedback": ["Bring Python project experience forward"],
                "resume_iteration": 2,
            })

        self.assertEqual(result["resume_text"], "Revised resume")
        self.assertEqual(result["resume_iteration"], 3)
        self.assertIn("Current draft", captured["human"])
        self.assertIn("Bring Python project experience forward", captured["human"])
        pdf_dir = Path(draft_node.__file__).resolve().parents[1] / "user_data"
        for pdf_path in pdf_dir.rglob("*.pdf"):
            self.assertIn(f"[Source resume: {pdf_path.name}]", captured["human"])


class LatexWorkflowTests(unittest.TestCase):
    def test_latex_node_uses_template_preamble_and_best_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "format.tex"
            template.write_text(
                r"\documentclass{article}" "\n" r"\begin{document}" "\n"
                "Example content that should be replaced\n" r"\end{document}" "\n",
                encoding="utf-8",
            )
            generated_body = r"\section{Skills}" "\n" r"\begin{resume_list}" "\n" r"\item Python" "\n" r"\end{resume_list}"

            def generate(prompt_value):
                if prompt_value.to_messages()[0].content.startswith(
                    "Give the job description a concise filename label"
                ):
                    return "Machine Learning Intern"
                return generated_body

            model = RunnableLambda(generate)
            with (
                patch.object(latex_node, "TEMPLATE_PATH", template),
                patch.object(latex_node, "OUTPUT_PATH", root / "generated"),
                patch.object(latex_node, "llm7_model", model),
            ):
                result = latex_node.resume_to_latex({
                    "best_resume_text": "Verified candidate resume",
                    "resume_text": "Lower scoring draft",
                    "jd": "Machine learning internship",
                })

            latex_path = Path(result["latex_path"])
            output = latex_path.read_text(encoding="utf-8")
            self.assertEqual(latex_path.name, "machine-learning-intern.tex")
            self.assertTrue(output.startswith(r"\documentclass{article}"))
            self.assertIn(generated_body, output)
            self.assertNotIn("Example content", output)
            self.assertEqual(output.count(r"\begin{document}"), 1)
            self.assertEqual(output.count(r"\end{document}"), 1)

    def test_latex_node_uses_a_unique_name_when_artifacts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "format.tex"
            template.write_text(
                r"\documentclass{article}" "\n" r"\begin{document}" "\n"
                r"\end{document}" "\n",
                encoding="utf-8",
            )
            output_dir = root / "generated"
            output_dir.mkdir()
            (output_dir / "machine-learning-intern.tex").write_text("existing tex")
            (output_dir / "machine-learning-intern-2.pdf").write_bytes(b"existing pdf")
            model = RunnableLambda(
                lambda prompt: (
                    "Machine Learning Intern"
                    if prompt.to_messages()[0].content.startswith(
                        "Give the job description a concise filename label"
                    )
                    else r"\section{Skills}"
                )
            )
            with (
                patch.object(latex_node, "TEMPLATE_PATH", template),
                patch.object(latex_node, "OUTPUT_PATH", output_dir),
                patch.object(latex_node, "llm7_model", model),
            ):
                result = latex_node.resume_to_latex({
                    "best_resume_text": "Verified candidate resume",
                    "jd": "Machine learning internship",
                })

            self.assertEqual(Path(result["latex_path"]).name, "machine-learning-intern-3.tex")
            self.assertEqual(
                (output_dir / "machine-learning-intern.tex").read_text(),
                "existing tex",
            )
            self.assertEqual(
                (output_dir / "machine-learning-intern-2.pdf").read_bytes(),
                b"existing pdf",
            )

    def test_filename_label_requires_two_or_three_words_and_is_sanitized(self) -> None:
        for label in ("Engineer", "Machine Learning Software Engineer"):
            with self.subTest(label=label), patch.object(
                latex_node,
                "llm7_model",
                RunnableLambda(lambda _, value=label: value),
            ):
                with self.assertRaisesRegex(ValueError, "2- or 3-word"):
                    latex_node._generate_filename_stem("A job description")

        with patch.object(
            latex_node,
            "llm7_model",
            RunnableLambda(lambda _: '"Machine Learning Intern!"'),
        ):
            self.assertEqual(
                latex_node._generate_filename_stem("A job description"),
                "machine-learning-intern",
            )

    def test_pdf_node_compiles_and_reports_missing_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            latex_file = Path(temp_dir) / "resume.tex"
            valid_source = r"\documentclass{article}\begin{document}Resume\end{document}"
            latex_file.write_text(valid_source)

            def create_pdf(*args, **kwargs):
                return self._compiler_result(args, returncode=0, create_pdf=True)

            with (
                patch.object(pdf_node.shutil, "which", return_value="pdflatex"),
                patch.object(pdf_node.subprocess, "run", side_effect=create_pdf),
            ):
                result = pdf_node.latex_to_pdf({"latex_path": str(latex_file)})
            self.assertTrue(Path(result["pdf_path"]).is_file())
            self.assertEqual(latex_file.read_text(), valid_source)

            with patch.object(pdf_node.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "pdflatex is required"):
                    pdf_node.latex_to_pdf({"latex_path": str(latex_file)})

    def test_pdf_node_repairs_extra_closing_brace_and_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            latex_file = Path(temp_dir) / "resume.tex"
            broken_source = (
                r"\documentclass{article}\begin{document}"
                r"\documentTitle{Candidate}{\faPhone\ 123}}"
                r"\end{document}"
            )
            corrected_source = (
                r"\documentclass{article}\begin{document}"
                r"\documentTitle{Candidate}{\faPhone\ 123}"
                r"\end{document}"
            )
            latex_file.write_text(broken_source)
            repair_calls = []

            def repair(prompt_value):
                repair_calls.append(prompt_value.to_messages())
                return AIMessage(content=f"```latex\n{corrected_source}\n```")

            compiler_calls = []

            def compile_with_repair(*args, **kwargs):
                compiler_calls.append(args[0])
                return self._compiler_result(
                    args,
                    returncode=1 if len(compiler_calls) == 1 else 0,
                    create_pdf=len(compiler_calls) > 1,
                )

            with (
                patch.object(pdf_node.shutil, "which", return_value="pdflatex"),
                patch.object(pdf_node.subprocess, "run", side_effect=compile_with_repair),
                patch.object(pdf_node, "llm7_model", RunnableLambda(repair)),
            ):
                result = pdf_node.latex_to_pdf({"latex_path": str(latex_file)})

            self.assertEqual(len(compiler_calls), 2)
            self.assertEqual(len(repair_calls), 1)
            self.assertIn("Too many }'s", repair_calls[0][1].content)
            self.assertIn(broken_source, repair_calls[0][1].content)
            self.assertEqual(latex_file.read_text(), corrected_source)
            self.assertTrue(Path(result["pdf_path"]).is_file())

    def test_pdf_node_stops_after_two_failed_repairs_and_keeps_source_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            latex_file = Path(temp_dir) / "resume.tex"
            broken_source = r"\documentclass{article}\begin{document}Bad }} source\end{document}"
            corrected_source = r"\documentclass{article}\begin{document}Still broken\end{document}"
            latex_file.write_text(broken_source)
            repair_count = 0
            compile_count = 0

            def repair(_):
                nonlocal repair_count
                repair_count += 1
                return AIMessage(content=corrected_source)

            def failing_compile(*args, **kwargs):
                nonlocal compile_count
                compile_count += 1
                return self._compiler_result(args, returncode=1, create_pdf=False)

            with (
                patch.object(pdf_node.shutil, "which", return_value="pdflatex"),
                patch.object(pdf_node.subprocess, "run", side_effect=failing_compile),
                patch.object(pdf_node, "llm7_model", RunnableLambda(repair)),
            ):
                with self.assertRaisesRegex(RuntimeError, "after 2 repair attempts"):
                    pdf_node.latex_to_pdf({"latex_path": str(latex_file)})

            self.assertEqual(repair_count, 2)
            self.assertEqual(compile_count, 3)
            self.assertEqual(latex_file.read_text(), corrected_source)
            self.assertIn("Too many }'s", latex_file.with_suffix(".log").read_text())

    def test_pdf_node_keeps_compiler_log_if_repair_request_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            latex_file = Path(temp_dir) / "resume.tex"
            original_source = r"\documentclass{article}\begin{document}Bad }} source\end{document}"
            latex_file.write_text(original_source)

            with (
                patch.object(pdf_node.shutil, "which", return_value="pdflatex"),
                patch.object(
                    pdf_node.subprocess,
                    "run",
                    side_effect=lambda *args, **kwargs: self._compiler_result(
                        args,
                        returncode=1,
                        create_pdf=False,
                    ),
                ),
                patch.object(pdf_node, "_repair_latex", side_effect=RuntimeError("LLM7 unavailable")),
            ):
                with self.assertRaisesRegex(RuntimeError, "LLM7 unavailable"):
                    pdf_node.latex_to_pdf({"latex_path": str(latex_file)})

            self.assertEqual(latex_file.read_text(), original_source)
            self.assertIn("Too many }'s", latex_file.with_suffix(".log").read_text())

    @staticmethod
    def _compiler_result(args, *, returncode: int, create_pdf: bool) -> subprocess.CompletedProcess[str]:
        command = args[0]
        output_dir = Path(command[command.index("-output-directory") + 1])
        (output_dir / "resume.log").write_text(
            "" if returncode == 0 else "! Too many }'s.\nl.1 Bad }} source\n",
            encoding="utf-8",
        )
        if create_pdf:
            (output_dir / "resume.pdf").write_bytes(b"%PDF-1.4")
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout="" if returncode == 0 else "! Too many }'s.",
            stderr="",
        )


if __name__ == "__main__":
    unittest.main()
