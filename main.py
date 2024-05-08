import streamlit as st

from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from langchain.agents import create_tool_calling_agent
from langchain import hub

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")

st.title("Test Chatbot")

openai_api_key = ""
os.environ["OPENAI_API_KEY"] = openai_api_key

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    search = DuckDuckGoSearchRun(name="Search")
    prompts = hub.pull("hwchase17/openai-functions-agent")

    search_agent = create_tool_calling_agent(llm, [search], prompts)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.invoke({
            "input": st.session_state.messages,
            "intermediate_steps": []
        }, {"callbacks": [st_cb]})
        print(response)
        st.session_state.messages.append({"role": "assistant", "content": response.return_values})
        st.write(response.return_values)
