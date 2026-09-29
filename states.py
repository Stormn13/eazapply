from typing import TypedDict, NotRequired

class ResumeState(TypedDict):
    jd_link : str
    error_code : NotRequired[int]
    raw_website_text : NotRequired[str]
    jd: NotRequired[str]
