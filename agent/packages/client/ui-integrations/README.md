---
description: "Settings → Integrations: add MCP servers from presets, test and sign in to them, create signed webhooks with a delivery log, and set up notification targets."
kind: "package-reference"
---

# @deepseek-ai/dsh-client-ui-integrations

## Summary

`dsh-client-ui-integrations` adds the **Integrations** section to Web Settings. Everything it shows and changes goes through the `integrations` Remote of [`dsh-integrations`](../../integrations/integrations/README.md).

**Connected services (MCP)**
- Each server card shows:
  - its live phase, tools with read-only / may-change badges, and its error;
  - call counts and the time of the last call;
  - the server's recent output;
  - Enabled, Reconnect, Sign in / Sign out, Edit and Remove.
- Editor:
  - transport, command and arguments or URL;
  - write-only secrets;
  - exposure, approval mode and sign-in;
  - per-tool checkboxes;
  - Test, which shows the server's stderr on failure.

**Telegram**
- Token, folder, presets and approval handling.
- Notifications about other chats.
- Live bot status.
- A pairing code with its `t.me` link, and the list of paired chats.

**Webhooks**
- Presets: GitHub issues, GitHub CI failures, Home Assistant / Zapier / IFTTT, custom.
- Auth mode (signature or token) and chat mode (new or same).
- The public base URL, and a one-time secret box with instructions for the chosen mode.
- The delivery log, with "Run again".

**Notifications**
- Targets with format, events and a minimum turn length.
- Test.

**Deep links**
- The plugin also opens `?session=<id>` (the links in notifications) once the session list has loaded, then removes the parameter.

The section polls the overview every 3 seconds while open, and reloads when an OAuth tab reports back.

## Model Experience

None: this package is UI only.

## Known Limitations and Deferred Work

- **Copy is English only.** The Chinese dictionary repeats the English copy.
- **Sign-in may replace the page.** When the browser blocks the sign-in popup, the page itself navigates to the provider.
