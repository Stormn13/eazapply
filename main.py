#imports
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import webscrap
from states import ResumeState
#this is the state schema
#define error codes
# 1 - the link for the jd doesn't exist


#graph stuff
graph = StateGraph(ResumeState)
#graph nodes
graph.add_node('webscrap', webscrap.webscrap)
#graph edges
graph.add_edge(START, 'webscrap')
graph.add_edge('webscrap', END)
#compliation and running
workflow = graph.compile()

initial_State: ResumeState = {
    'jd_link' : "https://www.linkedin.com/jobs/view/4471337817/",
    'error_code' : 0,
    'raw_website_text' : ""
}
final_state = workflow.invoke(initial_State)
print(final_state)

