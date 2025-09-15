from flat_mcp_client.tools import MCPToolbox

# to set up CRAWL4AI's official mcp server:
#   docker pull unclecode/crawl4ai:0.6.0rc1-r2
#   docker run -d -p 11235:11235 --name crawl4ai --shm-size=1g unclecode/crawl4ai:0.6.0rc1-r2

mcp_config = {
  "mcpServers": {
    "crawl4ai": {
      "type": "sse",
      "url": "http://localhost:11235/mcp/sse",
    }
  }
}

toolbox = MCPToolbox('crawl4ai', mcp_config)
