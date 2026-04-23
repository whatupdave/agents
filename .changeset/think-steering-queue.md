---
"@cloudflare/think": patch
---

Queue overlapping chat submits as steering during active Think inference loops, add `saveMessages(..., { concurrency: "steer" | "followUp" })`, inject steered messages into the next `streamText` step when possible, and fall back to running them as follow-up turns if the current loop finishes first.
