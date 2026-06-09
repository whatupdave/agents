import { describe, expect, it } from "vitest";
import { env } from "cloudflare:workers";
import { getAgentByName } from "agents";

async function freshSteeringAgent(name: string) {
  return getAgentByName(env.SteeringTestAgent, name);
}

type PromptMessage = {
  role: string;
  content: Array<Record<string, unknown>>;
};

function parsePrompt(prompt: string): PromptMessage[] {
  return JSON.parse(prompt) as PromptMessage[];
}

function userTexts(prompt: string): string[] {
  return parsePrompt(prompt)
    .filter((message) => message.role === "user")
    .map((message) => JSON.stringify(message.content));
}

describe("Think — saveMessages steering", () => {
  it("folds a mid-turn steer into the active turn at the next step", async () => {
    const agent = await freshSteeringAgent("steer-mid-turn");
    const run = await agent.runMidTurnSteer();

    // One turn, two model calls: tool step, then the (steered) final step.
    expect(run.prompts).toHaveLength(2);
    expect(JSON.stringify(userTexts(run.prompts[0]))).toContain("2pm");
    expect(JSON.stringify(userTexts(run.prompts[0]))).not.toContain("3pm");
    expect(JSON.stringify(userTexts(run.prompts[1]))).toContain("2pm");
    expect(JSON.stringify(userTexts(run.prompts[1]))).toContain("3pm");

    // The steered message enters the conversation as the LAST message —
    // after the in-flight assistant/tool steps, not spliced into history.
    const secondPrompt = parsePrompt(run.prompts[1]);
    const lastMessage = secondPrompt[secondPrompt.length - 1];
    expect(lastMessage.role).toBe("user");
    expect(JSON.stringify(lastMessage.content)).toContain("3pm");
    const toolIndex = secondPrompt.findIndex((m) => m.role === "tool");
    expect(toolIndex).toBeGreaterThan(-1);
    expect(toolIndex).toBeLessThan(secondPrompt.length - 1);

    // Both promises resolve against the SAME turn.
    expect(run.turn.status).toBe("completed");
    expect(run.steer.status).toBe("completed");
    expect(run.steer.steered).toBe(true);
    expect(run.steer.requestId).toBe(run.turn.requestId);

    // Single combined response: [user 2pm, user 3pm, assistant].
    expect(run.roles).toEqual(["user", "user", "assistant"]);
  });

  it("preserves in-turn reasoning provider metadata across a steered step", async () => {
    const agent = await freshSteeringAgent("steer-reasoning");
    const run = await agent.runMidTurnSteer();

    // The second step's prompt must still carry the first step's reasoning
    // part WITH its provider metadata (OpenAI Responses reasoning items live
    // only in the run's in-memory messages mid-turn — rebuilding from
    // session history would drop them).
    const secondPrompt = parsePrompt(run.prompts[1]);
    const assistant = secondPrompt.find((m) => m.role === "assistant");
    expect(assistant).toBeDefined();
    const reasoning = assistant?.content.find(
      (part) => part.type === "reasoning"
    );
    expect(reasoning).toBeDefined();
    expect(JSON.stringify(reasoning)).toContain("rs_1");
    expect(JSON.stringify(reasoning)).toContain("sig-1");
  });

  it("folds multiple steer calls into the same step and settles both", async () => {
    const agent = await freshSteeringAgent("steer-double");
    const run = await agent.runDoubleSteer();

    expect(run.prompts).toHaveLength(2);
    const secondUserTexts = JSON.stringify(userTexts(run.prompts[1]));
    expect(secondUserTexts).toContain("3pm");
    expect(secondUserTexts).toContain("standup");

    expect(run.steer.steered).toBe(true);
    expect(run.secondSteer?.steered).toBe(true);
    expect(run.steer.requestId).toBe(run.turn.requestId);
    expect(run.secondSteer?.requestId).toBe(run.turn.requestId);
    expect(run.roles).toEqual(["user", "user", "user", "assistant"]);
  });

  it("keeps a steered message in the model's view on every later step", async () => {
    const agent = await freshSteeringAgent("steer-reinjection");
    const run = await agent.runReinjectionSteer();

    // One turn, three model calls: tool, tool, final text. The steer drains
    // at the boundary before call 2 — and must STILL be in call 3's prompt.
    // The AI SDK rebuilds each step's input from the turn's initial messages
    // plus its own response messages, so a single-step prepareStep override
    // evaporates; the window re-appends injected messages every step.
    expect(run.prompts).toHaveLength(3);
    expect(JSON.stringify(userTexts(run.prompts[0]))).not.toContain("3pm");
    expect(JSON.stringify(userTexts(run.prompts[1]))).toContain("3pm");
    expect(JSON.stringify(userTexts(run.prompts[2]))).toContain("3pm");

    // Injected once — re-appending must not duplicate it within a prompt.
    const finalPromptUsers = userTexts(run.prompts[2]);
    expect(
      finalPromptUsers.filter((text) => text.includes("3pm"))
    ).toHaveLength(1);

    expect(run.steer.steered).toBe(true);
    expect(run.steer.requestId).toBe(run.turn.requestId);
    expect(run.roles).toEqual(["user", "user", "assistant"]);

    // The hook fired once, inside the host turn, with the steered message.
    expect(run.steeredBatches).toHaveLength(1);
    expect(run.steeredBatches[0]).toHaveLength(1);
  });

  it("combines steers that miss the final step into one fallback turn", async () => {
    const agent = await freshSteeringAgent("steer-batched-fallback");
    const run = await agent.runDoubleFallbackSteer();

    // Two model calls: the host turn, then ONE fallback turn carrying both
    // leftover steers — not a turn per entry re-answering the same history.
    expect(run.prompts).toHaveLength(2);
    const fallbackUserTexts = JSON.stringify(userTexts(run.prompts[1]));
    expect(fallbackUserTexts).toContain("3pm");
    expect(fallbackUserTexts).toContain("standup");

    expect(run.steer.status).toBe("completed");
    expect(run.secondSteer?.status).toBe("completed");
    expect(run.steer.steered).toBeUndefined();
    expect(run.secondSteer?.steered).toBeUndefined();
    expect(run.steer.requestId).toBe(run.secondSteer?.requestId);
    expect(run.steer.requestId).not.toBe(run.turn.requestId);

    // Nothing drained, so the hook never fired.
    expect(run.steeredBatches).toHaveLength(0);

    // [user 2pm, assistant, user 3pm, user standup, assistant].
    expect(run.roles).toEqual([
      "user",
      "assistant",
      "user",
      "user",
      "assistant"
    ]);
  });

  it("falls back to a queued turn when the steer arrives after the final step", async () => {
    const agent = await freshSteeringAgent("steer-post-final");
    const run = await agent.runPostFinalStepSteer();

    // Two model calls across two turns: the host turn never saw the steered
    // message; the fallback turn did.
    expect(run.prompts).toHaveLength(2);
    expect(JSON.stringify(userTexts(run.prompts[0]))).not.toContain("3pm");
    expect(JSON.stringify(userTexts(run.prompts[1]))).toContain("3pm");

    expect(run.turn.status).toBe("completed");
    expect(run.steer.status).toBe("completed");
    expect(run.steer.steered).toBeUndefined();
    expect(run.steer.requestId).not.toBe(run.turn.requestId);

    // Two separate responses: [user 2pm, assistant, user 3pm, assistant].
    expect(run.roles).toEqual(["user", "assistant", "user", "assistant"]);
  });

  it('drops a steer:"require" call that misses the active turn', async () => {
    const agent = await freshSteeringAgent("steer-require-post-final");
    const run = await agent.runPostFinalStepRequireSteer();

    // Only the host turn's model call — no fallback turn ran, and the
    // steered message was never persisted.
    expect(run.prompts).toHaveLength(1);
    expect(run.turn.status).toBe("completed");
    expect(run.steer.status).toBe("skipped");
    expect(run.steer.requestId).toBe("");
    expect(run.steer.steered).toBeUndefined();
    expect(run.roles).toEqual(["user", "assistant"]);
  });

  it('drops a steer:"require" call when no turn is active', async () => {
    const agent = await freshSteeringAgent("steer-require-idle");
    const run = await agent.runIdleRequireSteer();

    expect(run.prompts).toHaveLength(0);
    expect(run.steer.status).toBe("skipped");
    expect(run.steer.requestId).toBe("");
    expect(run.roles).toEqual([]);
  });

  it("behaves like a normal saveMessages call when no turn is active", async () => {
    const agent = await freshSteeringAgent("steer-idle");
    const run = await agent.runIdleSteer();

    expect(run.prompts).toHaveLength(1);
    expect(run.steer.status).toBe("completed");
    expect(run.steer.steered).toBeUndefined();
    expect(run.roles).toEqual(["user", "assistant"]);
  });

  it("refuses to steer non-user messages and queues them instead", async () => {
    const agent = await freshSteeringAgent("steer-non-user");
    const run = await agent.runNonUserSteer();

    // Host turn (2 calls) + the queued non-user turn (1 call).
    expect(run.prompts).toHaveLength(3);
    // The host turn's final step never saw the assistant note.
    expect(JSON.stringify(parsePrompt(run.prompts[1]))).not.toContain(
      "afternoon slots"
    );
    expect(run.steer.steered).toBeUndefined();
    expect(run.steer.requestId).not.toBe(run.turn.requestId);
  });
});
