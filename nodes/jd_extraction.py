from extras.states import ResumeState
from extras.models import llm7_model

def jd_extraction(state: ResumeState):
    raw_text = state['raw_website_text'] # type: ignore
    prompt = f"take this text of the full website - {raw_text} and explain the job description in consise manner and in bullet points"
    output = llm7_model.invoke(prompt).content
    return {'jd' : output}