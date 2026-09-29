from states import ResumeState
from models import geniric_model, googe_model

def jd_extraction(state: ResumeState):
    raw_text = state['raw_website_text'] # type: ignore
    prompt = f"take this text of the full website - {raw_text} and explain the job description in consise manner and in bullet points"
    output = geniric_model.invoke(prompt).content
    return {'jd' : output}