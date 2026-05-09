import asyncio
import json
import os
import threading
from pathlib import Path
from textwrap import dedent
from typing import Any, TypedDict
from bs4 import BeautifulSoup
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from selenium import webdriver
from sentence_transformers import CrossEncoder

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


MODEL_NAME = "qwen/qwen3-32b"
VECTOR_DB_DIR = Path.cwd() / "vector_db"
RAG_SCORE_THRESHOLD = 0.8
DEFAULT_MEMORY_WINDOW = 5
DEFAULT_WEB_URL = (
    "https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant"
)
DEFAULT_MCP_CONFIG_PATH = Path.cwd() / "mcp_servers.json"

TOOLS = dedent(
    """
    You have access to the following tools:

    1. scrape(url: str)
       - Use this to extract content from a website URL.

    2. explain(text: str)
       - Use this to explain or summarize given content.

    3. rag(query: str)
       - Use this for factual questions about indexed pages or stored knowledge.
       - Prefer this over explain when the user is asking a question about a scraped page.
       - It should answer only from retrieved chunks in the vector database.

    4. mcp_agent(query: str)
       - Use this when the user wants tools from an MCP server.
       - MCP servers can be local stdio processes or remote HTTP/SSE servers.

    Decide which tool to use based on the user input.
    If no tool is needed, return: finish
    """
).strip()


class AgentState(TypedDict):
    tools: str
    inputs: str
    last_task: str
    web_url: str
    scraped_content: str
    last_response: str
    task: str
    history: list[dict[str, str]]
    memory_window: int
    mcp_servers: dict[str, dict[str, Any]]


def require_groq_api_key() -> None:
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your environment before running the agent."
        )


_llm: ChatGroq | None = None
_embedding_model: HuggingFaceEmbeddings | None = None
_vectorstore: Chroma | None = None
_reranker: CrossEncoder | None = None
_semantic_splitter: SemanticChunker | None = None


def get_llm() -> ChatGroq:
    global _llm

    if _llm is None:
        require_groq_api_key()
        _llm = ChatGroq(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )
    return _llm


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def get_vectorstore() -> Chroma:
    global _vectorstore

    if _vectorstore is None:
        VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        _vectorstore = Chroma(
            collection_name="rag_docs",
            embedding_function=get_embedding_model(),
            persist_directory=str(VECTOR_DB_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vectorstore


def get_reranker() -> CrossEncoder:
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def get_semantic_splitter() -> SemanticChunker:
    global _semantic_splitter

    if _semantic_splitter is None:
        _semantic_splitter = SemanticChunker(get_embedding_model())
    return _semantic_splitter

window_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
)


def load_mcp_server_config(config_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load MCP server config from a JSON file or MCP_SERVERS_CONFIG env var."""
    path = Path(config_path) if config_path else DEFAULT_MCP_CONFIG_PATH
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            config = json.load(file)
    else:
        raw_config = os.getenv("MCP_SERVERS_CONFIG", "").strip()
        if not raw_config:
            return {}
        config = json.loads(raw_config)

    if not isinstance(config, dict):
        raise ValueError("MCP server config must be a JSON object keyed by server name.")
    return config


def format_mcp_error(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        messages = [format_mcp_error(inner) for inner in exc.exceptions]
        return "; ".join(message for message in messages if message)
    return str(exc) or exc.__class__.__name__


async def load_mcp_tools(
    servers: dict[str, dict[str, Any]],
) -> tuple[list[Any], list[str]]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    tools = []
    errors = []
    for server_name, server_config in servers.items():
        try:
            client = MultiServerMCPClient({server_name: server_config})
            tools.extend(await client.get_tools())
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            errors.append(f"{server_name}: {format_mcp_error(exc)}")
    return tools, errors


async def arun_mcp_agent(
    user_text: str,
    mcp_servers: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Run a LangChain agent with tools loaded from one or more MCP servers."""
    servers = mcp_servers or load_mcp_server_config()
    mcp_tools = []
    mcp_errors = []
    if servers:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "langchain-mcp-adapters is not installed. Install it with: "
                "pip install langchain-mcp-adapters"
            ) from exc

        mcp_tools, mcp_errors = await load_mcp_tools(servers)
        if not mcp_tools:
            return (
                "I could not load tools from the configured MCP server(s).\n\n"
                "Start the HTTP MCP server first:\n"
                "env\\Scripts\\python.exe mcp_scrapper_server.py\n\n"
                "Failed MCP servers:\n"
                + "\n".join(f"- {error}" for error in mcp_errors)
            )

    mcp_status = ""
    if mcp_errors:
        mcp_status = (
            "\n\nSome MCP servers failed to load:\n"
            + "\n".join(f"- {error}" for error in mcp_errors)
        )

    agent = create_agent(
        get_llm(),
        [scrape, explain, rag, *mcp_tools],
        system_prompt=dedent(
            """
            You are the MCP-capable version of the scraper agent.
            Use local scraper-agent tools and MCP server tools as one shared toolset.
            When MCP server tools are available, prefer them over local tools.
            For a request to scrape and explain a URL, prefer scrape_and_explain_url.
            Use scrape to fetch webpages, rag for questions about indexed scraped pages,
            and explain for plain-language summaries.
            """
        ).strip(),
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_text}]}
    )
    messages = result.get("messages", [])
    if not messages:
        return mcp_status.strip()
    return f"{getattr(messages[-1], 'content', str(messages[-1]))}{mcp_status}"


def run_mcp_agent(
    user_text: str,
    mcp_servers: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Synchronous wrapper for the async MCP LangChain agent."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(arun_mcp_agent(user_text, mcp_servers))

    result: dict[str, str | BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(arun_mcp_agent(user_text, mcp_servers))
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return str(result.get("value", ""))


def scrape_website(url: str) -> str:
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        html = driver.page_source
    finally:
        driver.quit()

    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def explain_content(text: str) -> str:
    prompt = dedent(
        f"""
        Explain the following website content in simple terms:

        {text}
        """
    ).strip()
    return get_llm().invoke(prompt).content


def rerank(
    query: str,
    docs_with_scores: list[tuple[Document, float]],
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    docs = [doc for doc, _ in docs_with_scores]
    pairs = [(query, doc.page_content) for doc in docs]
    scores = get_reranker().predict(pairs)
    reranked = list(zip(docs, scores, strict=False))
    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked[:top_k]


@tool
def scrape(url: str) -> str:
    """Scrape a webpage and return its content."""
    return scrape_website(url)


@tool
def explain(text: str) -> str:
    """Explain given text content."""
    return explain_content(text)


@tool
def rag(query: str, url_filter: str | None = None) -> str:
    """Answer a question using the local RAG vector store with reranking."""
    search_kwargs: dict[str, object] = {"k": 20}
    if url_filter:
        search_kwargs["filter"] = {"url": url_filter}

    vectorstore = get_vectorstore()
    matches = vectorstore.similarity_search_with_score(query, **search_kwargs)
    if not matches:
        return "I don't know based on the indexed data."

    relevant_matches = [
        (doc, score) for doc, score in matches if score <= RAG_SCORE_THRESHOLD
    ]
    if not relevant_matches:
        return "I don't know based on the indexed data."

    reranked_matches = rerank(query, relevant_matches, top_k=5)
    context_lines = []
    for rank, (doc, score) in enumerate(reranked_matches, start=1):
        source_url = doc.metadata.get("url", "unknown")
        context_lines.append(
            f"Source {rank} | url: {source_url} | score: {score:.4f}\n"
            f"{doc.page_content}"
        )

    context_text = "\n\n".join(context_lines)
    prompt = dedent(
        f"""
        You are a strict RAG assistant.

        Answer ONLY using the context below.
        If the answer is not clearly present, say:
        "I don't know based on the indexed data."

        Question:
        {query}

        Context:
        {context_text}

        Answer:
        """
    ).strip()
    return get_llm().invoke(prompt).content


def scrape_node(state: AgentState) -> dict[str, str]:
    content = scrape.invoke({"url": state["web_url"]})
    window_chunks = window_splitter.split_text(content)

    docs = []
    for chunk in window_chunks:
        sem_docs = get_semantic_splitter().create_documents([chunk])
        docs.extend(sem_docs)

    for doc in docs:
        doc.metadata = {"url": state["web_url"], "source": "web_scrape"}

    vectorstore = get_vectorstore()
    vectorstore.delete(where={"url": state["web_url"]})
    vectorstore.add_documents(docs)

    return {
        "scraped_content": content,
        "last_response": f"Sliding window + semantic chunking -> {len(docs)} chunks.",
    }


def explain_node(state: AgentState) -> dict[str, str]:
    explanation = explain.invoke({"text": state["scraped_content"]})
    print(explanation)
    return {"last_response": explanation}


def rag_node(state: AgentState) -> dict[str, str]:
    answer = rag.invoke({"query": state["inputs"], "url_filter": state.get("web_url")})
    print(answer)
    return {"last_response": answer}


def mcp_node(state: AgentState) -> dict[str, str]:
    mcp_servers = state.get("mcp_servers") or load_mcp_server_config()
    state["mcp_servers"] = mcp_servers
    answer = run_mcp_agent(state["inputs"], mcp_servers)
    print(answer)
    return {"last_response": answer}


def rule_based_task(state: AgentState) -> str:
    user_text = state["inputs"].strip().lower()
    has_scraped_content = bool(state.get("scraped_content"))
    has_url_context = bool(state.get("web_url"))
    last_task = state.get("last_task", "")

    asks_history = any(
        phrase in user_text
        for phrase in [
            "previous",
            "earlier",
            "last response",
            "what did you say",
            "from history",
            "before",
        ]
    )
    asks_scrape = any(
        phrase in user_text for phrase in ["scrape", "fetch", "load", "index", "crawl"]
    )
    asks_explain = any(
        phrase in user_text
        for phrase in ["explain", "summarize", "summary", "describe"]
    )
    asks_mcp = any(
        phrase in user_text
        for phrase in [
            "mcp",
            "mcp agent",
            "server tool",
            "external tool",
            "use tool",
            "tools from",
        ]
    )

    question_starts = (
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "compare",
        "difference",
    )
    is_question = "?" in user_text or user_text.startswith(question_starts)

    if asks_history:
        return "recall"
    if asks_mcp:
        return "mcp"
    if asks_explain and last_task == "scrape" and has_scraped_content:
        return "explain"
    if asks_scrape and last_task == "scrape":
        return "finish"
    if asks_scrape and last_task != "scrape":
        return "scrape"
    if is_question and (has_scraped_content or has_url_context):
        return "rag"
    if asks_explain and has_scraped_content:
        return "explain"
    return ""


def brain(state: AgentState) -> dict[str, str]:
    rule_choice = rule_based_task(state)
    if rule_choice:
        return {"task": rule_choice, "last_task": rule_choice}

    formatted_history = "\n".join(
        f"User: {turn['user']} | Agent: {turn['response']}"
        for turn in state["history"]
    )
    prompt = dedent(
        f"""
        You are an agent. Decisions: scrape, explain, rag, recall, mcp, finish.
        History: {formatted_history}
        MCP servers configured: {bool(state.get('mcp_servers'))}
        Input: {state['inputs']}
        Last Task: {state['last_task']}
        Return one word only.
        """
    ).strip()
    raw_decision = get_llm().invoke(prompt).content.strip().lower()
    decision = raw_decision.split()[0] if raw_decision else "finish"

    if decision not in {"scrape", "explain", "rag", "recall", "mcp", "finish"}:
        decision = "finish"

    return {"task": decision, "last_task": decision}


def recall_node(state: AgentState) -> dict[str, str]:
    formatted_history = []
    for turn in state["history"]:
        formatted_history.append(
            "User: {user} | Task: {task} | Agent: {response}".format(
                user=turn.get("user", ""),
                task=turn.get("task", ""),
                response=turn.get("response", ""),
            )
        )
    formatted_history_text = "\n".join(formatted_history)

    prompt = dedent(
        f"""
        You answer questions using the conversation memory below.

        Conversation memory:
        {formatted_history_text}

        Current user question:
        {state['inputs']}

        Answer only from the stored memory. If the answer is not present, say you do not have that earlier response in memory.
        """
    ).strip()

    answer = get_llm().invoke(prompt).content.strip()
    print(answer)
    return {"last_response": answer}


def route_task(state: AgentState) -> str:
    print(f"{state['task']} called\n")
    return state["task"]


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("brain", brain)
    builder.add_node("scrape", scrape_node)
    builder.add_node("explain", explain_node)
    builder.add_node("rag", rag_node)
    builder.add_node("recall", recall_node)
    builder.add_node("mcp", mcp_node)

    builder.set_entry_point("brain")
    builder.add_conditional_edges(
        "brain",
        route_task,
        {
            "scrape": "scrape",
            "explain": "explain",
            "rag": "rag",
            "recall": "recall",
            "mcp": "mcp",
            "finish": END,
        },
    )
    builder.add_edge("scrape", "brain")
    builder.add_edge("explain", END)
    builder.add_edge("rag", END)
    builder.add_edge("recall", END)
    builder.add_edge("mcp", END)

    return builder.compile()


graph = build_graph()

state: AgentState = {
    "inputs": "",
    "tools": TOOLS,
    "web_url": DEFAULT_WEB_URL,
    "scraped_content": "",
    "last_response": "",
    "task": "",
    "history": [],
    "last_task": "",
    "memory_window": DEFAULT_MEMORY_WINDOW,
    "mcp_servers": load_mcp_server_config(),
}


def run_agent(
    user_text: str,
    web_url: str | None = None,
    mcp_servers: dict[str, dict[str, Any]] | None = None,
) -> AgentState:
    global state

    state["inputs"] = user_text
    if web_url is not None:
        state["web_url"] = web_url
    if mcp_servers is not None:
        state["mcp_servers"] = mcp_servers

    result = graph.invoke(state)
    state.update(result)

    executed_task = state.get("task", "finish")
    state["last_task"] = executed_task
    state["history"].append(
        {
            "user": user_text,
            "task": executed_task,
            "response": state.get("last_response", ""),
        }
    )

    window = state.get("memory_window", DEFAULT_MEMORY_WINDOW)
    state["history"] = state["history"][-window:]

    return state


def main() -> None:
    print("Scrapper agent ready. Type 'exit' or 'quit' to stop.")
    print(f"Current URL: {state['web_url']}")
    print(f"MCP servers configured: {', '.join(state['mcp_servers']) or 'none'}")

    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break

        web_url = input("URL override (blank to keep current): ").strip() or None
        result = run_agent(user_text, web_url)
        response = result.get("last_response") or "Finished."
        print(f"\nAgent: {response}")


if __name__ == "__main__":
    main()
