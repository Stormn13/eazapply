from extras.states import ResumeState
from extras.models import llm7_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path


def format_docs(documents):
    return "\n\n".join(
        f"[Source resume: {Path(doc.metadata.get('source', 'resume')).name}]\n{doc.page_content}"
        for doc in documents
        if doc.page_content.strip()
    )


def jd_to_resume(state: ResumeState):
    data_path = Path(__file__).resolve().parents[1] / "user_data"
    docs = [
        doc
        for path in sorted(data_path.rglob("*.pdf"))
        for doc in PyPDFLoader(str(path)).load()
        if doc.page_content.strip()
    ]
    if not docs:
        raise FileNotFoundError(f"No readable resume PDFs found under {data_path}")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Rewrite a resume using only facts explicitly stated in the provided source resumes. "
            "Do not invent or infer the candidate's name, contact details, education, dates, GPA, "
            "employers, job titles, projects, technologies, achievements, or metrics. Omit details "
            "that are not supported by the source resumes; never use placeholders. Tailor the "
            "selection and wording to the job description, but do not claim a skill or experience "
            "unless a source resume supports it. Return only the finished resume. Do not include "
            "an introduction, analysis, match table, assumptions, or follow-up advice."
        ),
        (
            "human",
            "Job description:\n{job_description}\n\n"
            "Source resumes:\n{source_resumes}\n\n"
            "Produce one concise, job-tailored resume based on these source resumes."
        ),
    ])

    chain = (
        prompt
        | llm7_model
        | StrOutputParser()
    )

    response = chain.invoke({
        "source_resumes": format_docs(docs),
        "job_description": state["jd"],  # type: ignore
    })

    return {"resume_text": response}
