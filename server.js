const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const { StreamableHTTPServerTransport } = require('@modelcontextprotocol/sdk/server/streamableHttp.js');
const { CallToolRequestSchema, ListToolsRequestSchema } = require('@modelcontextprotocol/sdk/types.js');
const OpenAI = require('openai');
const http = require('http');
const { similaritySearch, getCategories, getChunkCount, setupSchema } = require('./db.js');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function embed(text) {
  const resp = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  return resp.data[0].embedding;
}

function createMcpServer() {
  const server = new Server(
    { name: 'acs-rag-mcp', version: '1.0.0' },
    { capabilities: { tools: {} } }
  );

  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: 'search_acs',
        description: 'Semantically search the Azure Communication Services REST API reference (1,123 pages). Returns the most relevant documentation chunks for any question about ACS endpoints, parameters, authentication, WhatsApp/Advanced Messaging, SMS, Chat, Call Automation, Identity, Email, Phone Numbers, or Rooms.',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: 'Natural language query, e.g. "how to send a WhatsApp message", "HMAC authentication", "provision phone number", "send SMS", "create chat thread"',
            },
            top_k: {
              type: 'number',
              description: 'Number of results to return (default: 5, max: 10)',
              default: 5,
            },
            category: {
              type: 'string',
              description: 'Optional: filter by category. Options: advanced-messaging, sms, chat, call-automation, identity, email, phone-numbers, rooms, network-traversal, auth, general',
            },
          },
          required: ['query'],
        },
      },
      {
        name: 'list_acs_categories',
        description: 'List all available ACS API categories in the knowledge base with chunk counts.',
        inputSchema: { type: 'object', properties: {}, required: [] },
      },
      {
        name: 'acs_stats',
        description: 'Get stats about the ACS knowledge base (total chunks indexed, coverage).',
        inputSchema: { type: 'object', properties: {}, required: [] },
      },
    ],
  }));

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    if (name === 'search_acs') {
      const { query, top_k = 5, category } = args;
      const k = Math.min(top_k, 10);
      const queryEmbedding = await embed(query);
      const results = await similaritySearch(queryEmbedding, k, category || null);

      if (!results.length) {
        return { content: [{ type: 'text', text: 'No results found for that query.' }] };
      }

      const formatted = results.map((r, i) => {
        const score = (r.score * 100).toFixed(1);
        const cat = r.category ? ` [${r.category}]` : '';
        return `### Result ${i + 1}${cat} — ${score}% match\n\n${r.content}`;
      });

      return { content: [{ type: 'text', text: formatted.join('\n\n---\n\n') }] };
    }

    if (name === 'list_acs_categories') {
      const cats = await getCategories();
      if (!cats.length) {
        return { content: [{ type: 'text', text: 'No categories found — knowledge base may not be ingested yet.' }] };
      }
      const lines = cats.map(c => `- **${c.category || 'uncategorized'}**: ${c.chunk_count} chunks`);
      return { content: [{ type: 'text', text: lines.join('\n') }] };
    }

    if (name === 'acs_stats') {
      const count = await getChunkCount();
      return {
        content: [{
          type: 'text',
          text: `ACS RAG knowledge base:\n- Total chunks indexed: ${count.toLocaleString()}\n- Embedding model: text-embedding-3-small (1536 dims)\n- Source: Azure Communication Services REST API reference (1,123 pages)`,
        }],
      };
    }

    return { content: [{ type: 'text', text: `Unknown tool: ${name}` }], isError: true };
  });

  return server;
}

async function main() {
  await setupSchema();

  const PORT = process.env.PORT;

  if (PORT) {
    const httpServer = http.createServer(async (req, res) => {
      if (req.method === 'GET' && req.url === '/health') {
        const count = await getChunkCount();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'ok', chunks: count }));
        return;
      }

      if (req.method === 'POST' && req.url === '/mcp') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', async () => {
          try {
            const parsedBody = JSON.parse(body);
            const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
            const server = createMcpServer();
            await server.connect(transport);
            await transport.handleRequest(req, res, parsedBody);
          } catch (err) {
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: err.message }));
          }
        });
        return;
      }

      res.writeHead(404);
      res.end('Not found');
    });

    httpServer.listen(PORT, () => {
      console.log(`ACS RAG MCP HTTP server on port ${PORT}`);
      console.log(`MCP endpoint: POST /mcp`);
      console.log(`Health check: GET /health`);
    });
  } else {
    const server = createMcpServer();
    const transport = new StdioServerTransport();
    await server.connect(transport);
    process.stderr.write('ACS RAG MCP stdio server running\n');
  }
}

main().catch(err => {
  process.stderr.write(`Fatal: ${err.message}\n`);
  process.exit(1);
});
