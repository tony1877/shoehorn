---
description: "Runtime-configured MCP servers, a two-way Telegram bot, inbound webhooks, and outbound notifications for the Web settings page, with secrets in the credential store."
kind: "package-reference"
---

# @deepseek-ai/dsh-integrations

## Summary

`dsh-integrations` connects the agent to the outside world from **Settings → Integrations**, without editing `cordis.yml` or restarting:

- **MCP servers.** One `dsh-mcp-client` child per enabled row, restarted live when the row or its secrets change.
- **Telegram bot.** Talk to the agent from Telegram, with text and photos, and approve tool calls or answer its questions with buttons.
- **Inbound webhooks.** `POST /hooks/<id>` starts a chat, or continues one, with a prompt filled from the body.
- **Outbound notifications.** Formatted, signed POSTs when a top-level chat finishes, needs approval, or asks a question. They can also go to paired Telegram chats.

Where things are kept:

| What | Where |
|---|---|
| Non-secret settings | the `integrations` settings namespace |
| Secrets: MCP environment variables, headers and OAuth grants; webhook secrets; notification URLs and signing secrets; the bot token | credential-store grant records under the `integrations` scope (the page never reads the values back) |
| Chat bindings, the Telegram update offset, the webhook delivery log | JSON state files in `$DSH_HOME/integrations/` |

## Use this package

Mount it on a Web Host that also mounts settings, credentials, the web server and the session services (the `web-app` bundle does):

```yaml
- id: integrations
  name: '@deepseek-ai/dsh-integrations'
```

The optional `config` takes:

- `servers`, `webhooks`, `notifications`, `telegram` and `publicBaseUrl`, as the composition base the user's edits resolve above;
- `stateDirectory`, the folder for the state files;
- `telegramApiBase`, a self-hosted Telegram Bot API server.

### MCP servers

- **Transports.** A row is a local program (`stdio`) or a URL (`streamable-http`, or `sse` for older servers).
- **Exposure.**
  - `native` registers every tool.
  - `deferred` (the default) registers two proxies instead:
    - `mcp_search(query)` returns matching tools with their schemas. Its description lists each service's tool names.
    - `mcp_call(tool, arguments)` runs one.
- **Tool allowlist.** `includeTools` limits the tools offered.
- **Approval.** `approval` is `ask`, `ask-writes` (the default; a tool is allowed only when the server marks it `readOnlyHint` and not `destructiveHint`), or `allow`.
  - It is enforced in `tools/pre-execute` for direct and proxied calls alike.
  - A session whose approval policy is `never` (full access) is not asked.
- **OAuth.** A row with `auth: oauth` waits as "Needs sign-in" until a grant exists.
  - **Sign in** runs discovery, dynamic client registration and PKCE, and returns to `/integrations/oauth/callback`.
  - Tokens refresh on later connections.
- **Diagnostics.** The page shows each server's phase and last error.
  - The last 60 lines of a stdio server's stderr (Test failures include the tail).
  - Calls, failures and the time of the last call.
- **Presets.**
  - Playwright browser: headless, isolated, with snapshots written as absolute paths under `$DSH_HOME/browser-output`.
  - Context7 docs.
  - GitHub's hosted server.
  - Notion (OAuth).
  - A filesystem server.

### Telegram

1. Create a bot with @BotFather and paste its token.
2. Set the folder Telegram chats work in, and turn the bot on.
3. Create a pairing code on the page, and send `/start <code>` to the bot, or open the `t.me` link the page shows. Codes last ten minutes.

How the bot works:

- **Connection.** It uses long polling, so it needs no public URL. The update offset is persisted, so a restart neither loses nor repeats messages. A new token starts from offset 0.
- **Access.** Only paired chats are served. An unknown chat is told how to pair, at most once an hour.
- **One chat per Telegram chat.** Each paired chat talks to its own dsh chat, which is resumed from its log after a restart.
  - `/new` starts a new chat.
  - `/stop` cancels the running turn.
  - `/status` shows the current chat.
  - `/help` lists the commands.
- **Messages.**
  - Text and photos (or image documents) are accepted. An image's type is taken from its bytes, and it reaches the model as an image attachment.
  - Voice notes, stickers and other files are declined with a message.
- **Answers.**
  - At the end of each turn, the turn's newest assistant text is sent as Telegram HTML. Markdown is converted, and long answers are split under the size limit without breaking code blocks. A piece Telegram cannot parse is resent as plain text.
  - "typing…" shows while the agent works.
  - A turn that wrote no new text sends nothing.
- **Approvals and questions.** With `answerInTelegram` on, approvals and questions for a Telegram chat are claimed by the bot.
  - An approval shows the tool, the reason and the arguments, with Allow / Deny buttons. A call through `mcp_call` shows the tool it runs.
  - A question shows its options as buttons (multi-select has a Done button), or takes a typed reply.
  - A withdrawn request is marked "no longer needed".
- **Notifications about other chats.** `notifyEvents` sends these to every paired chat; `minTurnSeconds` applies to "finished".

### Webhooks

Senders authenticate in one of two ways:

| Mode | Sender sends |
|---|---|
| `hmac` | `sha256=<hex HMAC-SHA256 of the raw body>` in the signature header (GitHub's `X-Hub-Signature-256` works as is) |
| `token` | the secret as `Authorization: Bearer <secret>`, or `?token=<secret>` (Home Assistant, Zapier, IFTTT, curl) |

| Response | When |
|---|---|
| 401 | authentication failed; logged without the body |
| 404 | unknown or disabled endpoint |
| 405 | not a POST |
| 413 | body over `maxBodyBytes` |
| 429 | over `ratePerMinute` |
| 503 | no secret is set |
| 202 | accepted; processing continues after the response |

- **Filter.** `matchPath`/`matchValues` decide whether a delivery acts.
- **Templates.** `promptTemplate` and `titleTemplate` fill `{{dot.path}}` and `{{body}}`. An empty title falls back to the endpoint's title, then its id.
- **Chat mode.** `chatMode: new` starts a chat per delivery; `same` continues the endpoint's chat, resuming it if needed.
- **Delivery log.** The last 100 deliveries are kept in `webhooks.json`. Bodies up to 64 KB are kept, so "Run again" works after a restart.
- **Presets.** GitHub new issues (triage), GitHub failed CI runs, Home Assistant / Zapier / IFTTT (token mode, same chat), and custom signed JSON.

### Notifications

| Format | Body |
|---|---|
| `json` | `{event, sessionId, title, text, link?, turnSeconds?}`, signed in `X-Signature-256` |
| `discord` | `{content}` |
| `slack` | `{text}` |
| `ntfy` | plain text, with `title`, `tags` and `click` headers |

- **Skipping quick turns.** "Chat finished" is skipped for turns shorter than the target's `minTurnSeconds` (default 20), which you were most likely watching.
- **Links.** Links are `<publicBaseUrl>/?session=<id>`; the web client opens that chat.
- **Delivery.** Sends never block the agent. Up to three attempts are made with backoff. Client errors other than 429 are not retried.

## Model Experience

### What the model sees

- **Native rows:** their tools.
- **Deferred rows:**
  - `mcp_search`, whose description names each connected service and its tools, and whose results carry descriptions and JSON input schemas;
  - `mcp_call`, whose failures append a hint to look up the schema.
- **Telegram chats:** messages arrive as ordinary user messages (source `user`); photos arrive as image blocks.
- **Webhook chats:** the filled prompt arrives as a user message sourced `webhook`.

### Token effect

- A deferred service costs one line per service in the `mcp_search` description, instead of every schema in every request.
- Search results enter context only when the model asks.
- Telegram photos cost image tokens like any attached image.

### KV Cache effect

- The proxy descriptions change only when the set of deferred tools changes. The prefix then changes from the tool definitions onward.
- Telegram and webhook follow-ups append to their chat.

## Known Limitations and Deferred Work

- **Images through the proxy.** Images returned through `mcp_call` reach the model only when the underlying definition's `finalizeContent` accepts the proxy execution. Otherwise they are described as text.
- **Telegram claims approvals.** While `answerInTelegram` is on, approvals and questions for a Telegram chat are answered in Telegram only. The web page shows no card for them.
- **Telegram groups.** In a paired group, every member can talk to the agent and press its buttons.
- **Untrusted content.** Content from MCP results, web pages, Telegram or webhook bodies is not marked untrusted for later approval decisions.
- **Approval default.** A session without an explicit approval-policy event falls back to `ask` semantics even if the deployment's default is `never`.
