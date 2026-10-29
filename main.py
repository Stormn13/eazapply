#imports
from langgraph.graph import StateGraph, START, END
import nodes.webscrap as webscrap
import nodes.jd_extraction as jd_extraction
import nodes.jd_to_resume as jd_to_resume
import nodes.ats_score as ats_score
import nodes.resume_to_latex as resume_to_latex
import nodes.latex_to_pdf as latex_to_pdf
from extras.states import ResumeState
#this is the state schema
#define error codes
# 1 - the link for the jd doesn't exist


#graph stuff
graph = StateGraph(ResumeState)
#graph nodes
graph.add_node('webscrap', webscrap.webscrap)
graph.add_node('jd_extraction', jd_extraction.jd_extraction)
graph.add_node('jd_to_resume', jd_to_resume.jd_to_resume)
graph.add_node('ats_score', ats_score.ats_score)
graph.add_node('resume_to_latex', resume_to_latex.resume_to_latex)
graph.add_node('latex_to_pdf', latex_to_pdf.latex_to_pdf)
#graph edges
graph.add_edge(START, 'webscrap')
graph.add_edge('webscrap', 'jd_extraction')
graph.add_edge('jd_extraction', 'jd_to_resume')
graph.add_edge('jd_to_resume', 'ats_score')
graph.add_conditional_edges(
    'ats_score',
    ats_score.route_after_ats_score,
    {
        'revise_resume': 'jd_to_resume',
        'generate_latex': 'resume_to_latex',
    },
)
graph.add_edge('resume_to_latex', 'latex_to_pdf')
graph.add_edge('latex_to_pdf', END)
#compliation and running
workflow = graph.compile()

if __name__ == "__main__":
    initial_State: ResumeState = {
        'jd_link' : "https://www.linkedin.com/jobs/view/4471337817/",
    }
    final_state = workflow.invoke(initial_State)
    print(f"Resume PDF created: {final_state['pdf_path']}")
