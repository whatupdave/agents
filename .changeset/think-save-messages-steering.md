---
"@cloudflare/think": minor
"agents": patch
---

Add mid-turn steering to `Think.saveMessages` via `{ steer: true }`. When a chat turn is already running, steered user messages are persisted and injected into the active inference loop at the next step boundary, so the model adjusts course and produces a single response covering both the original request and the follow-up ("put a block at 2pm" → "actually 3pm" no longer yields two responses). The injection appends to the run's in-flight model messages, preserving in-turn provider state such as OpenAI Responses reasoning items across steps. The steered call's promise resolves with the host turn's outcome and `steered: true`. Steering falls back to today's queued-turn behavior when no turn is active, the active turn is a structured workflow turn, the messages arrive after the model's final step, or any message is not `role: "user"`. A new `chat:steered` observability event is emitted when messages are folded into an active turn.
