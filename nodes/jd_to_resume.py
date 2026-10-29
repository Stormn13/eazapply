from extras.states import ResumeState
from extras.models import googe_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


def jd_to_resume(state: ResumeState):
    DATA_PATH = "user_data"
    docs = [
        doc
        for path in Path(DATA_PATH).glob("*.pdf")
        for doc in PyPDFLoader(str(path)).load()
    ]

    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the following resumes to restructure it according to the given job description:\n\n{context}"),
    ("human", "{question}")
    ])

    chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | googe_model
    | StrOutputParser()
    )

    response = chain.invoke({
    "context": format_docs(docs),
    "question": f"provide a resume with all the given documents and the job description {state['jd']}" # type: ignore
    })

    return {"resume_text" : response}

