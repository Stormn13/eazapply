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
            model = RunnableLambda(lambda _: generated_body)
            with (
                patch.object(latex_node, "TEMPLATE_PATH", template),
                patch.object(latex_node, "OUTPUT_PATH", root / "generated"),
                patch.object(latex_node, "llm7_model", model),
            ):
                result = latex_node.resume_to_latex({
                    "best_resume_text": "Verified candidate resume",
                    "resume_text": "Lower scoring draft",
                })

            output = Path(result["latex_path"]).read_text(encoding="utf-8")
            self.assertTrue(output.startswith(r"\documentclass{article}"))
            self.assertIn(generated_body, output)
            self.assertNotIn("Example content", output)
            self.assertEqual(output.count(r"\begin{document}"), 1)
            self.assertEqual(output.count(r"\end{document}"), 1)

    def test_pdf_node_compiles_and_reports_missing_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            latex_file = Path(temp_dir) / "resume.tex"
            latex_file.write_text(r"\documentclass{article}\begin{document}Resume\end{document}")

            def create_pdf(*args, **kwargs):
                output_dir = Path(args[0][args[0].index("-output-directory") + 1])
                (output_dir / "resume.pdf").write_bytes(b"%PDF-1.4")
                return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

            with (
                patch.object(pdf_node.shutil, "which", return_value="pdflatex"),
                patch.object(pdf_node.subprocess, "run", side_effect=create_pdf),
            ):
                result = pdf_node.latex_to_pdf({"latex_path": str(latex_file)})
            self.assertTrue(Path(result["pdf_path"]).is_file())

            with patch.object(pdf_node.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "pdflatex is required"):
                    pdf_node.latex_to_pdf({"latex_path": str(latex_file)})


if __name__ == "__main__":
    unittest.main()
