/**
 * Test agent for saveMessages steering (`{ steer: true }`).
 *
 * Uses mock LanguageModelV3 implementations that gate mid-turn so the test
 * can inject steered messages at deterministic points: while a tool is
 * executing (drained at the next prepareStep) and while the final text is
 * streaming (after the last prepareStep — the leftover/fallback path).
 */

import type { LanguageModel, ToolSet, UIMessage } from "ai";
import { tool } from "ai";
import { z } from "zod";
import { Think } from "../../think";
import type { ThinkSaveMessagesResult } from "../../think";

const v3FinishReason = (unified: "stop" | "tool-calls") => ({
  unified,
  raw: undefined
});
const v3Usage = (inputTokens: number, outputTokens: number) => ({
  inputTokens: {
    total: inputTokens,
    noCache: inputTokens,
    cacheRead: 0,
    cacheWrite: 0
  },
  outputTokens: { total: outputTokens, text: outputTokens, reasoning: 0 }
});

function userMessage(text: string): UIMessage {
  return {
    id: crypto.randomUUID(),
    role: "user",
    parts: [{ type: "text", text }]
  };
}

type SteeringRunSummary = {
  /** Full LanguageModelV3 prompt per model call, JSON-serialized. */
  prompts: string[];
  turn: ThinkSaveMessagesResult;
  steer: ThinkSaveMessagesResult;
  secondSteer?: ThinkSaveMessagesResult;
  /** Roles of persisted history, in order. */
  roles: string[];
};

export class SteeringTestAgent extends Think {
  private _mode: "tool-then-text" | "gated-text" = "tool-then-text";
  private _modelCalls = 0;
  private _prompts: string[] = [];
  private _toolStarted: (() => void) | null = null;
  private _toolGate: (() => void) | null = null;
  private _streamStarted: (() => void) | null = null;
  private _streamGate: (() => void) | null = null;

  getModel(): LanguageModel {
    return this._mode === "tool-then-text"
      ? this._createToolThenTextModel()
      : this._createGatedTextModel();
  }

  getTools(): ToolSet {
    return {
      wait: tool({
        description: "Wait for the test gate to release",
        inputSchema: z.object({}),
        execute: async () => {
          const gate = new Promise<void>((resolve) => {
            this._toolGate = resolve;
          });
          this._toolStarted?.();
          await gate;
          return "waited";
        }
      })
    };
  }

  /**
   * Steer while the tool is executing — the message must be folded into the
   * active turn at the next prepareStep and answered by the same response.
   */
  async runMidTurnSteer(): Promise<SteeringRunSummary> {
    this._reset("tool-then-text");
    const toolStarted = new Promise<void>((resolve) => {
      this._toolStarted = resolve;
    });
    const turnPromise = this.saveMessages([
      userMessage("create a calendar block at 2pm")
    ]);
    await toolStarted;
    const steerPromise = this.saveMessages(
      [userMessage("actually make it 3pm")],
      { steer: true }
    );
    this._toolGate?.();
    const [turn, steer] = await Promise.all([turnPromise, steerPromise]);
    return this._summarize(turn, steer);
  }

  /**
   * Two steer calls during the same tool execution — both fold into the
   * active turn and settle with its outcome.
   */
  async runDoubleSteer(): Promise<SteeringRunSummary> {
    this._reset("tool-then-text");
    const toolStarted = new Promise<void>((resolve) => {
      this._toolStarted = resolve;
    });
    const turnPromise = this.saveMessages([
      userMessage("create a calendar block at 2pm")
    ]);
    await toolStarted;
    const steerPromise = this.saveMessages(
      [userMessage("actually make it 3pm")],
      { steer: true }
    );
    const secondSteerPromise = this.saveMessages(
      [userMessage("and title it standup")],
      { steer: true }
    );
    this._toolGate?.();
    const [turn, steer, secondSteer] = await Promise.all([
      turnPromise,
      steerPromise,
      secondSteerPromise
    ]);
    return { ...this._summarize(turn, steer), secondSteer };
  }

  /**
   * Steer while the final (and only) text step is already streaming — past
   * the last prepareStep, so the window closes with the entry undrained and
   * it must fall back to its own queued turn.
   */
  async runPostFinalStepSteer(): Promise<SteeringRunSummary> {
    this._reset("gated-text");
    const streamStarted = new Promise<void>((resolve) => {
      this._streamStarted = resolve;
    });
    const turnPromise = this.saveMessages([
      userMessage("create a calendar block at 2pm")
    ]);
    await streamStarted;
    const steerPromise = this.saveMessages(
      [userMessage("actually make it 3pm")],
      { steer: true }
    );
    this._streamGate?.();
    const [turn, steer] = await Promise.all([turnPromise, steerPromise]);
    return this._summarize(turn, steer);
  }

  /** Steer with no turn active — must behave exactly like saveMessages. */
  async runIdleSteer(): Promise<SteeringRunSummary> {
    this._reset("gated-text");
    // Release the gate as soon as the stream starts; nothing to steer into.
    this._streamStarted = () => this._streamGate?.();
    const steer = await this.saveMessages(
      [userMessage("create a calendar block at 2pm")],
      { steer: true }
    );
    return this._summarize(steer, steer);
  }

  /**
   * Steer with a non-user message while a turn is active — steering is
   * refused and the call runs as its own queued turn.
   */
  async runNonUserSteer(): Promise<SteeringRunSummary> {
    this._reset("tool-then-text");
    const toolStarted = new Promise<void>((resolve) => {
      this._toolStarted = resolve;
    });
    const turnPromise = this.saveMessages([
      userMessage("create a calendar block at 2pm")
    ]);
    await toolStarted;
    const assistantNote: UIMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      parts: [{ type: "text", text: "noted: prefer afternoon slots" }]
    };
    const steerPromise = this.saveMessages([assistantNote], { steer: true });
    this._toolGate?.();
    const [turn, steer] = await Promise.all([turnPromise, steerPromise]);
    return this._summarize(turn, steer);
  }

  private _reset(mode: "tool-then-text" | "gated-text"): void {
    this._mode = mode;
    this._modelCalls = 0;
    this._prompts = [];
    this._toolStarted = null;
    this._toolGate = null;
    this._streamStarted = null;
    this._streamGate = null;
  }

  private _summarize(
    turn: ThinkSaveMessagesResult,
    steer: ThinkSaveMessagesResult
  ): SteeringRunSummary {
    return {
      prompts: this._prompts,
      turn,
      steer,
      roles: this.messages.map((message) => message.role)
    };
  }

  private _recordPrompt(options: Record<string, unknown>): void {
    this._prompts.push(
      JSON.stringify((options as { prompt?: unknown[] }).prompt ?? [])
    );
  }

  /**
   * Call 1: reasoning (with provider metadata, like OpenAI Responses
   * reasoning items) followed by a `wait` tool call. Later calls: plain
   * text. Keys off the model-call count so a steered second step does not
   * change which step it is on.
   */
  private _createToolThenTextModel(): LanguageModel {
    return {
      specificationVersion: "v3",
      provider: "test",
      modelId: "steering-tool-model",
      supportedUrls: {},
      doGenerate() {
        throw new Error("doGenerate not implemented in mock");
      },
      doStream: (options: Record<string, unknown>) => {
        this._modelCalls++;
        this._recordPrompt(options);
        const currentCall = this._modelCalls;
        const stream = new ReadableStream({
          start(controller) {
            controller.enqueue({ type: "stream-start", warnings: [] });
            if (currentCall === 1) {
              controller.enqueue({ type: "reasoning-start", id: "r1" });
              controller.enqueue({
                type: "reasoning-delta",
                id: "r1",
                delta: "planning the calendar block"
              });
              controller.enqueue({
                type: "reasoning-end",
                id: "r1",
                providerMetadata: {
                  test: { itemId: "rs_1", signature: "sig-1" }
                }
              });
              controller.enqueue({
                type: "tool-call",
                toolCallId: "tc1",
                toolName: "wait",
                input: JSON.stringify({})
              });
              controller.enqueue({
                type: "finish",
                finishReason: v3FinishReason("tool-calls"),
                usage: v3Usage(10, 5)
              });
            } else {
              controller.enqueue({ type: "text-start", id: `t${currentCall}` });
              controller.enqueue({
                type: "text-delta",
                id: `t${currentCall}`,
                delta: `Response ${currentCall}`
              });
              controller.enqueue({ type: "text-end", id: `t${currentCall}` });
              controller.enqueue({
                type: "finish",
                finishReason: v3FinishReason("stop"),
                usage: v3Usage(10, 5)
              });
            }
            controller.close();
          }
        });
        return Promise.resolve({ stream });
      }
    } as LanguageModel;
  }

  /**
   * Call 1: starts streaming text, signals the test, then parks until the
   * gate is released — keeping the turn past its last prepareStep. Later
   * calls complete immediately.
   */
  private _createGatedTextModel(): LanguageModel {
    return {
      specificationVersion: "v3",
      provider: "test",
      modelId: "steering-gated-text-model",
      supportedUrls: {},
      doGenerate() {
        throw new Error("doGenerate not implemented in mock");
      },
      doStream: (options: Record<string, unknown>) => {
        this._modelCalls++;
        this._recordPrompt(options);
        const currentCall = this._modelCalls;
        const agent = this;
        const stream = new ReadableStream({
          async start(controller) {
            controller.enqueue({ type: "stream-start", warnings: [] });
            controller.enqueue({ type: "text-start", id: `t${currentCall}` });
            controller.enqueue({
              type: "text-delta",
              id: `t${currentCall}`,
              delta: `Response ${currentCall}`
            });
            if (currentCall === 1) {
              const gate = new Promise<void>((resolve) => {
                agent._streamGate = resolve;
              });
              agent._streamStarted?.();
              await gate;
            }
            controller.enqueue({ type: "text-end", id: `t${currentCall}` });
            controller.enqueue({
              type: "finish",
              finishReason: v3FinishReason("stop"),
              usage: v3Usage(10, 5)
            });
            controller.close();
          }
        });
        return Promise.resolve({ stream });
      }
    } as LanguageModel;
  }
}
