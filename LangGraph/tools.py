from typing import Annotated, List, Tuple, Union

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools.arxiv.tool import ArxivQueryRun

duckduckgo = DuckDuckGoSearchRun(max_results=5)

# This executes code locally, which can be unsafe
python_repl_tool = PythonREPLTool()
arxiv_tool = ArxivQueryRun()


