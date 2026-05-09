import argparse
from mcp.server.fastmcp import FastMCP
from scrapper_agent import explain_content, rag, scrape_website

mcp = FastMCP(
    "scrapper-agent-tools",
    host="127.0.0.1",
    port=8000,
    streamable_http_path="/mcp",
)


@mcp.tool(name="scrape_url")
def scrape_url(url: str) -> str:
    """Scrape a webpage URL and return its visible text content."""
    return scrape_website(url)


@mcp.tool(name="explain_text")
def explain_text(text: str) -> str:
    """Explain or summarize text in simple terms."""
    return explain_content(text)


@mcp.tool(name="scrape_and_explain_url")
def scrape_and_explain_url(url: str) -> str:
    """Scrape a webpage URL and explain the content in simple terms."""
    content = scrape_website(url)
    return explain_content(content)


@mcp.tool(name="rag_answer")
def rag_answer(query: str, url_filter: str | None = None) -> str:
    """Answer a question using the scraper agent's local vector store."""
    return rag.invoke({"query": query, "url_filter": url_filter})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the scraper-agent MCP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
