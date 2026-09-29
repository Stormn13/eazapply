#imports
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import nodes.webscrap as webscrap
import nodes.jd_extraction as jd_extraction
from extras.states import ResumeState
#this is the state schema
#define error codes
# 1 - the link for the jd doesn't exist


#graph stuff
graph = StateGraph(ResumeState)
#graph nodes
graph.add_node('webscrap', webscrap.webscrap)
graph.add_node('jd_extraction', jd_extraction.jd_extraction)
#graph edges
graph.add_edge(START, 'webscrap')
graph.add_edge('webscrap', 'jd_extraction')
graph.add_edge('jd_extraction', END)
#compliation and running
workflow = graph.compile()

initial_State: ResumeState = {
    'jd_link' : "https://www.linkedin.com/jobs/view/4471337817/",
}
final_state = workflow.invoke(initial_State)
print(final_state["jd"])

