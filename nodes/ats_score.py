import json
from typing import Any

from extras.models import llm7_model
from extras.states import ResumeState
from langchain_core.prompts import ChatPromptTemplate


def _parse_ats_result(content: Any) -> tuple[float, list[str]]:
    if not isinstance(content, str):
        raise ValueError("ATS evaluator must return a JSON string.")

    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("ATS evaluator returned invalid JSON.") from exc

    if not isinstance(result, dict):
        raise ValueError("ATS evaluator response must be a JSON object.")

    score = result.get("score")
    feedback = result.get("feedback")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 10:
        raise ValueError("ATS evaluator score must be a number between 0 and 10.")
    if not isinstance(feedback, list) or not all(
        isinstance(item, str) and item.strip() for item in feedback
    ):
        raise ValueError("ATS evaluator feedback must be a list of non-empty strings.")

    return float(score), feedback


def ats_score(state: ResumeState) -> dict[str, object]:
    resume = state.get("resume_text")
    job_description = state.get("jd")
    if not resume or not job_description:
        raise ValueError("ATS scoring requires both resume_text and jd.")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Evaluate this resume against the job description as an ATS-focused reviewer. "
            "Score from 0 to 10 based on evidence-backed alignment of required skills, relevant "
            "experience/projects, keyword coverage, and clarity/readability. Do not reward "
            "fabricated or unsupported claims. Give concise, actionable improvements grounded "
            "only in facts already present in the resume. Return only valid JSON with exactly "
            'these keys: {{"score": number, "feedback": ["actionable improvement"]}}. '
            "Use an empty feedback array when there is no meaningful improvement to suggest."
        ),
        (
            "human",
            "Job description:\n{job_description}\n\nResume to evaluate:\n{resume}",
        ),
    ])
    response = (prompt | llm7_model).invoke({
        "job_description": job_description,
        "resume": resume,
    })
    score, feedback = _parse_ats_result(response.content)

    history = [*state.get("ats_score_history", []), score]
    best_score = state.get("best_ats_score", -1.0)
    best_resume = state.get("best_resume_text", "")
    if score > best_score:
        best_score = score
        best_resume = resume

    return {
        "ats_score": score,
        "ats_feedback": feedback,
        "ats_score_history": history,
        "best_ats_score": best_score,
        "best_resume_text": best_resume,
    }


ATS_TARGET_SCORE = 9.5
ATS_MIN_IMPROVEMENT = 0.1
ATS_MAX_ITERATIONS = 5


def route_after_ats_score(state: ResumeState) -> str:
    history = state.get("ats_score_history", [])
    if not history:
        raise ValueError("ATS routing requires a score history.")

    current_score = history[-1]
    iteration = state.get("resume_iteration", 0)
    reached_target = current_score >= ATS_TARGET_SCORE
    no_meaningful_improvement = len(history) > 1 and (
        current_score - history[-2] < ATS_MIN_IMPROVEMENT
    )
    reached_iteration_cap = iteration >= ATS_MAX_ITERATIONS

    if reached_target or no_meaningful_improvement or reached_iteration_cap:
        return "generate_latex"
    return "revise_resume"
