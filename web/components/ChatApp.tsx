"use client";

import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import {
  type Chat,
  type Message,
  createChat,
  deleteChat,
  getChat,
  listChats,
  sendMessage,
} from "@/lib/api";
import Markdown from "@/components/Markdown";
import MathInput from "@/components/MathInput";
import ThemeToggle from "@/components/ThemeToggle";
import { cn } from "@/lib/cn";

// URLs are "/" (new chat, created on the first message) and "/<chat id>". Switching chats
// uses history.pushState, which Next.js syncs into usePathname without a page navigation,
// so the sidebar and any in-flight answer stay mounted. Back/forward and reloads work too.
const go = (path: string, replace = false) =>
  replace ? window.history.replaceState(null, "", path) : window.history.pushState(null, "", path);

export default function ChatApp() {
  const chatId = usePathname().split("/")[1] || null;
  // Set by send() when it creates a chat: its messages are already in state, don't reload.
  // (Not "the loaded chat": React dev mode runs effects twice on mount, and the cancelled
  // first run would mark the chat loaded so the second run skipped it, leaving it empty.)
  const justCreated = useRef<string | null>(null);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const refreshChats = () => listChats().then(setChats).catch((e) => setError(String(e)));

  useEffect(() => {
    refreshChats();
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  // Load the chat named in the URL whenever it changes.
  useEffect(() => {
    if (chatId && chatId === justCreated.current) {
      justCreated.current = null;
      return;
    }
    abort.current?.abort();
    setMessages([]);
    setError(null);
    if (!chatId) return;
    let cancelled = false;
    getChat(chatId)
      .then((c) => !cancelled && setMessages(c.messages))
      .catch((e: Error) => {
        if (cancelled) return;
        setError(/^(404|422) /.test(e.message) ? "Chat not found." : e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [chatId]);

  // Plain clicks switch in place; cmd/ctrl/middle-click still open a new tab.
  const navigate = (path: string) => (e: React.MouseEvent) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    go(path);
  };

  async function removeChat(id: string) {
    if (!confirm("Delete this chat?")) return;
    try {
      await deleteChat(id);
    } catch (e) {
      return setError(String(e));
    }
    setChats((cs) => cs.filter((c) => c.id !== id));
    if (id === chatId) go("/");
  }

  // Update the last message (the assistant answer being streamed).
  const updateLast = (f: (m: Message) => Message) =>
    setMessages((ms) => [...ms.slice(0, -1), f(ms[ms.length - 1])]);

  async function send(e?: React.FormEvent) {
    e?.preventDefault();
    const content = input.trim();
    if (!content || streaming) return;

    let id = chatId;
    if (!id) {
      try {
        id = (await createChat()).id;
      } catch (e) {
        return setError(String(e));
      }
      justCreated.current = id;
      go(`/${id}`, true); // replace "/" so Back doesn't return to an empty chat
    }
    setInput("");
    setError(null);
    setStreaming(true);
    setMessages((ms) => [
      ...ms,
      { id: null, role: "user", content, sources: null },
      { id: null, role: "assistant", content: "", sources: null },
    ]);

    abort.current = new AbortController();
    await sendMessage(
      id,
      content,
      {
        onSources: (sources) => updateLast((m) => ({ ...m, sources })),
        onDelta: (text) => updateLast((m) => ({ ...m, content: m.content + text })),
        onError: (detail) => setError(detail),
        onDone: (messageId) => updateLast((m) => ({ ...m, id: messageId })),
      },
      abort.current.signal,
    ).catch((e: Error) => {
      if (e.name !== "AbortError") setError(String(e));
    });
    setStreaming(false);
    refreshChats(); // new title / ordering
  }

  return (
    <div className="flex h-dvh">
      <aside className="flex w-64 shrink-0 flex-col border-r border-base-300 bg-base-200 p-3">
        <a className="btn btn-sm btn-outline border-base-300 bg-base-100 w-full" href="/" onClick={navigate("/")}>
          + New chat
        </a>
        <ul className="menu mt-3 w-full flex-1 flex-nowrap overflow-y-auto p-0 [--menu-active-bg:var(--color-base-300)] [--menu-active-fg:var(--color-base-content)]">
          {chats.map((c) => (
            <li key={c.id} className="group relative">
              <a
                href={`/${c.id}`}
                onClick={navigate(`/${c.id}`)}
                title={c.title ?? undefined}
                aria-current={c.id === chatId ? "page" : undefined}
                className="block w-full truncate pr-8"
              >
                {c.title ?? "New chat"}
              </a>
              <button
                className="btn btn-ghost btn-xs btn-square absolute top-1/2 right-1 -translate-y-1/2 text-base-content/50 opacity-0 group-hover:opacity-100 hover:text-error focus-visible:opacity-100"
                onClick={() => removeChat(c.id)}
                title="Delete"
                aria-label="Delete chat"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
        <ThemeToggle className="mt-3" />
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {messages.length === 0 && (
            <p className="mt-[30vh] text-center text-base-content/50">Ask a question about hep-th papers.</p>
          )}
          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="chat chat-end mx-auto mb-3 max-w-chat">
                <div className="chat-bubble markdown max-w-[85%] bg-base-300 text-base-content">
                  <Markdown text={m.content} />
                </div>
              </div>
            ) : (
              <div key={i} className="mx-auto mb-6 max-w-chat">
                <div className="markdown">
                  {m.content ? (
                    <Markdown text={m.content} />
                  ) : (
                    streaming &&
                    i === messages.length - 1 && (
                      <span className="inline-flex items-center gap-2.5 text-sm text-base-content/60" role="status">
                        <span className="loading loading-dots loading-sm text-primary" aria-hidden="true" />
                        {m.sources === null ? "Searching papers" : "Thinking"}
                      </span>
                    )
                  )}
                </div>
                {m.sources && m.sources.length > 0 && (
                  <details className="mt-2 text-[13px] text-base-content/70">
                    <summary className="cursor-pointer select-none hover:text-base-content">
                      {m.sources.length} sources
                    </summary>
                    <ol className="mt-2 list-decimal space-y-2 pl-6">
                      {m.sources.map((s) => (
                        <li key={s.n}>
                          {s.url ? (
                            <a href={s.url} target="_blank" rel="noreferrer" className="link link-primary link-hover font-medium">
                              <Markdown text={s.title} inline />
                            </a>
                          ) : (
                            <span className="font-medium text-base-content">
                              <Markdown text={s.title} inline />
                            </span>
                          )}{" "}
                          <span className="text-xs text-base-content/50">
                            {s.authors} (arXiv:{s.paper_id}), <Markdown text={s.section} inline />
                          </span>
                          <div className="mt-0.5 leading-normal">
                            <Markdown text={s.snippet} inline />
                          </div>
                        </li>
                      ))}
                    </ol>
                  </details>
                )}
              </div>
            ),
          )}
          {error && (
            <div role="alert" className="alert alert-error alert-soft mx-auto max-w-chat">
              {error}
            </div>
          )}
          <div ref={bottom} />
        </div>

        <form
          onSubmit={send}
          className="mx-auto flex w-full max-w-[calc(var(--container-chat)+3rem)] items-end gap-2 border-t border-base-300 px-6 py-3"
        >
          <MathInput
            value={input}
            onChange={setInput}
            onSubmit={() => send()}
            placeholder="Ask about hep-th papers… LaTeX welcome: $\chi_i\chi_j$ (Shift+Enter: new line)"
          />
          {streaming ? (
            <button type="button" className="btn" onClick={() => abort.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="submit" className="btn btn-primary" disabled={!input.trim()}>
              Send
            </button>
          )}
        </form>
      </main>
    </div>
  );
}
