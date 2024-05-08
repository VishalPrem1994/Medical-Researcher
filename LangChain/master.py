from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

tools = [
    DuckDuckGoSearchRun(max_results=1)
]



prompt = hub.pull("hwchase17/react")
openai_api_key = ""
llm = ChatOpenAI(model="gpt-3.5-turbo-0125",openai_api_key = openai_api_key)

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(agent_executor.invoke({"input": "what is LangChain?"}))



