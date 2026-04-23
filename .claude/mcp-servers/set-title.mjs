#!/usr/bin/env node
//
// MCP server that sets the tmux / Ghostty terminal title.
// Uses official @modelcontextprotocol/sdk for reliable Claude Code integration.
//
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { openSync, writeSync, closeSync } from "node:fs";

function writeTty(data) {
  let fd;
  try {
    fd = openSync("/dev/tty", "w");
    writeSync(fd, data);
  } catch {
    process.stderr.write(data);
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

function setTitle(title) {
  writeTty(`\x1b]2;${title}\x07`);
  if (process.env.TMUX) {
    writeTty(`\x1bk${title}\x1b\\`);
  }
}

const server = new McpServer({
  name: "set-title",
  version: "0.1.0",
});

server.tool(
  "set_title",
  "Set the terminal tab/window title (works in Ghostty, tmux, iTerm2, and most xterm-compatible terminals)",
  {
    title: {
      type: "string",
      description: "The title to display on the terminal tab or window",
    },
  },
  async ({ title }) => {
    setTitle(title);
    return { content: [{ type: "text", text: `Title set to: ${title}` }] };
  }
);

const transport = new StdioServerTransport();
await server.connect(transport);
