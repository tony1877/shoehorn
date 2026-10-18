---
description: "Settings → Integrations: add MCP servers from presets, test and sign in to them, create signed webhooks with a delivery log, and set up notification targets."
kind: "package-reference"
---

# @deepseek-ai/dsh-client-ui-integrations

## Summary

`dsh-client-ui-integrations` adds the **Integrations** section to Web Settings. Everything it shows and changes goes through the `integrations` Remote of [`dsh-integrations`](../../integrations/integrations/README.md).

**Connected services (MCP)**
- Each server card shows its live phase, its tools with read-only and can-delete badges, and any error.
- Buttons: Enabled, Reconnect, Sign in / Sign out, Edit, Remove.
- Editor:
  - transport, command and arguments or URL;
  - secrets written in place, never read back;
  - exposure, approval mode, sign-in;
  - per-tool checkboxes;
  - Test, which connects once and lists the tools.
- Presets fill the editor.

**Webhooks**
- Public base URL.
- Endpoint cards with their full URL.
- A one-time secret box on creation or rotation.
- The delivery log, with "Run again".

**Notifications**
- Target cards with format, events and last send status.
- Test sends a sample message.
- A JSON target's signing secret is shown once.

The section polls the overview every 3 seconds while open, and reloads when an OAuth tab reports back.

## Model Experience

None: this package is UI only.

## Known Limitations and Deferred Work

- **Copy is English only.** The Chinese dictionary repeats the English copy.
- **Sign-in may replace the page.** When the browser blocks the sign-in popup, the page itself navigates to the provider.
