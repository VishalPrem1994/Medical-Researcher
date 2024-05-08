import operator
import time
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict
import functools
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from utils import create_agent, agent_node, search_agent_node
from tools import duckduckgo, python_repl_tool, arxiv_tool
from supervisor import supervisor_chain, members, llm

## IMPORTANT FOR RUNNING ASYNC FUNCTIONALITY
import nest_asyncio

nest_asyncio.apply()


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str


search_agent = create_agent(llm, [duckduckgo],
                            "You are a search agent. You have access to a search engine to find information on the web. Your job is to find news articles related to a list of topics provided along with links for each.")
search_node = functools.partial(search_agent_node, agent=search_agent, name="Searcher")

research_agent = create_agent(llm, [arxiv_tool],
                              "You are a researcher. You have access to a research database to find detailed information on the latest research. Your output will be a list of research papers with links which have good citations  for the question provided. "
                              "The format u should follow is to provide the title, authors, abstract, and link to the paper and then the next paper."
                              )
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

code_agent = create_agent(
    llm,
    [python_repl_tool],
    "You may generate safe python code to carry out simple tasks if needed"
)
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

editor_agent = create_agent(llm, [python_repl_tool],
                            """You are a professional  markdown editor. Your job is the take the results from the other workers 
                            and format them into a final report using markdown. Write code to write the final report to a markdown file locally.
                            """
                            )
editor_node = functools.partial(agent_node, agent=editor_agent, name="Editor")

workflow = StateGraph(AgentState)
workflow.add_node("Searcher", search_node)
workflow.add_node("Researcher", research_node)
workflow.add_node("Coder", code_node)
workflow.add_node("Editor", editor_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()
print(graph)

for s in graph.stream(
        {"messages": [HumanMessage(
            content="Get the details of the top research paper with the most citiations in medical research in the past year and news on the internet related to them with links.")]},
        {"recursion_limit": 10},
):
    if "__end__" not in s:
        print(s)
        print("----")

# graph.invoke(
#     {"messages": [HumanMessage(content="Write a brief research report on pikas.")]},
#     {"recursion_limit": 10},
# )
