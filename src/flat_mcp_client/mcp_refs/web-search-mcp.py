# Installation instructions at: https://github.com/mrkrsl/web-search-mcp
# NodeJS v22 seems to be required

mcp_config = {
  "mcpServers": {
    "web-search-mcp": {
      "command": "node",
      "args": ["../../web-search-mcp/dist/index.js"],
      "env": {
          "MAX_BROWSERS": "1",
      }
    }
  }
}
