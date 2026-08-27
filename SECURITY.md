# Security

## Reporting a vulnerability

Please open a [private security advisory](https://github.com/labatt/transcriber-studio/security/advisories/new)
rather than a public issue. A first response usually takes a few days — this is a side project, not
a staffed product.

## What this app handles

- **API keys** — for ElevenLabs, HuggingFace, and whichever LLM provider you enable. They are
  stored in `%APPDATA%\TranscriberStudio\settings.json` **in plain text**, readable by anything
  running as your user. That is the same trust model as most desktop tools, but it is worth knowing
  before you put a high-value key in. Use scoped or restricted keys where the provider offers them.
- **Recordings and transcripts** — audio, transcripts and glossaries are all local. Glossaries in
  particular accumulate real names and internal terminology, so treat an exported glossary as
  sensitive.
- **Outbound traffic** — nothing leaves your machine unless you enable it: the PLAUD CLI for its own
  account, ElevenLabs if you pick that engine, your chosen LLM provider if you enable AI Cleanup,
  and PyPI/npm/GitHub when you ask the Components window to check for updates.

## Installing and updating

The Components window can run `pip`, `npm` and `winget` on your behalf. Every command is shown in
full and confirmed before it runs, and an elevated run is a separate, explicit choice. If you would
rather not let an app modify your environment, copy the command and run it yourself — the button is
a convenience, not the only path.
