---
description: "Runtime-configured MCP servers, inbound webhooks, and outbound notifications for the Web settings page, with secrets in the credential store."
kind: "package-reference"
---

# @deepseek-ai/dsh-integrations

## Summary

`dsh-integrations` lets a person connect the agent to outside services from **Settings → Integrations** without editing `cordis.yml` or restarting:

- **MCP servers.** One `dsh-mcp-client` child per enabled row, restarted live when the row or its secrets change.
- **Inbound webhooks.** `POST /hooks/<id>`, HMAC-signed, starts a chat whose prompt is filled from the body.
- **Outbound notifications.** A signed or formatted POST when a top-level chat finishes, waits for approval, or asks a question.

Non-secret settings live in the `integrations` settings namespace. Secrets live in credential-store grant records under the `integrations` scope:

- MCP environment variables, headers and OAuth grants;
- webhook secrets;
- notification URLs and signing secrets.

The page reads names and flags only, never values.

## Use this package

Mount it on a Web Host that also mounts settings, credentials, the web server and the session services (the `web-app` bundle does):

```yaml
- id: integrations
  name: '@deepseek-ai/dsh-integrations'
```

The optional `config` takes `servers`, `webhooks`, `notifications`, and `publicBaseUrl` as the composition base. The user's edits resolve above it.

### MCP servers

- **Transports.** A row is either a local program (`stdio`) or a URL (`streamable-http`, or `sse` for older servers).
- **Exposure.**
  - `native` registers every tool as a model-visible tool.
  - `deferred` (the default) keeps the schemas out of the request and registers two proxies:
    - `mcp_search(query)` returns matching tools with their input schemas. Its description lists every connected service and its tool names.
    - `mcp_call(tool, arguments)` runs one.
- **Tool allowlist.** `includeTools` limits which tools are offered at all.
- **Approval.** `approval` decides when a call needs the user's approval: `ask`, `ask-writes` (the default), or `allow`.
  - `ask-writes` allows a tool only when the server marks it `readOnlyHint` and not `destructiveHint`.
  - The gate runs in `tools/pre-execute` for direct and proxied calls alike.
  - A session whose approval policy is `never` (full access) is not asked.
- **OAuth.** A row with `auth: oauth` is held, shown as "Needs sign-in", until a grant exists.
  - **Sign in** runs discovery, dynamic client registration and PKCE.
  - The provider returns to `/integrations/oauth/callback` on the origin the browser used.
  - The grant is stored, and the SDK refreshes tokens on later connections.
- **Presets.** The page offers presets for:
  - a Playwright browser (headless, isolated, absolute output paths under `$DSH_HOME/browser-output`);
  - Context7 docs;
  - GitHub's hosted server;
  - Notion (OAuth);
  - a filesystem server.

### Webhooks

Each endpoint has its own secret, shown once on creation or rotation. Senders sign the raw body like GitHub does:

```text
X-Signature-256: sha256=<hex HMAC-SHA256 of the body>
```

| Response | When |
|---|---|
| 401 | missing or wrong signature |
| 404 | unknown or disabled endpoint |
| 405 | not a POST |
| 413 | body over `maxBodyBytes` |
| 429 | over `ratePerMinute` |
| 503 | no secret is set |
| 202 | accepted; processing continues after the response |

- **Filter.** `matchPath`/`matchValues` decide whether the delivery starts a chat.
- **Templates.** `promptTemplate` and `titleTemplate` fill `{{dot.path}}` placeholders and `{{body}}`.
- **Delivery log.** The last 100 deliveries are kept in memory, with outcome, status, detail, session id and a body preview. Accepted ones can be run again.
- **Public base URL.** Only the URLs the page shows depend on it. Reaching the Host from the internet needs a tunnel (Cloudflare Tunnel, Tailscale Funnel).

### Notifications

| Format | Body |
|---|---|
| `json` | `{event, sessionId, title, text, link?}`, signed in `X-Signature-256` with the target's secret |
| `discord` | `{content}` |
| `slack` | `{text}` |
| `ntfy` | plain text, with `title`, `tags` and `click` headers |

- Sends never block the agent.
- Up to three attempts are made with backoff. Client errors other than 429 are not retried.

## Model Experience

### What the model sees

- **Native rows:** their tools, as `mcp-client` describes them.
- **Deferred rows:** `mcp_search` and `mcp_call`.
  - The `mcp_search` description names each connected service and its tools.
  - Search results carry each tool's description and JSON input schema.
  - `mcp_call` failures append a hint to look the schema up.
- **Webhook chats:** they start with the filled prompt as an ordinary user message, sourced `webhook`.

### Token effect

- A deferred service costs one line per service in the `mcp_search` description, instead of every tool schema in every request. For the browser preset that is about 25 schemas.
- Search results enter context only when the model asks.

### KV Cache effect

The proxy descriptions change only when the set of deferred tools changes (a server connects, disconnects, or changes its list). The prefix then changes from the tool definitions onward.

## Known Limitations and Deferred Work

- **Images through the proxy.** Images returned through `mcp_call` reach the model only when the underlying definition's `finalizeContent` accepts the proxy execution. Otherwise they are described as text.
- **The delivery log is in memory** and clears on restart.
- **Untrusted content.** Content from MCP results, web pages or webhook bodies is not yet marked untrusted for later approval decisions.
- **Approval default.** A session without an explicit approval-policy event falls back to `ask` semantics even if the deployment's default is `never`.
