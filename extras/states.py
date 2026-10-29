from typing import TypedDict, NotRequired

class ResumeState(TypedDict):
    jd_link : str
    error_code : NotRequired[int]
    raw_website_text : NotRequired[str]
    jd: NotRequired[str]
    resume_text: NotRequired[str]
    resume_iteration: NotRequired[int]
    ats_score: NotRequired[float]
    ats_feedback: NotRequired[list[str]]
    ats_score_history: NotRequired[list[float]]
    best_resume_text: NotRequired[str]
    best_ats_score: NotRequired[float]
    latex_path: NotRequired[str]
    pdf_path: NotRequired[str]