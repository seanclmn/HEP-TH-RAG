"use client";

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

export default function Home() {
  const [chats, setChats] = useState<Chat[]>([]);
  const [chatId, setChatId] = useState<string | null>(null);
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

  async function openChat(id: string) {
    abort.current?.abort();
    setChatId(id);
    setError(null);
    try {
      setMessages((await getChat(id)).messages);
    } catch (e) {
      setError(String(e));
    }
  }

  async function newChat() {
    abort.current?.abort();
    const chat = await createChat();
    setChats((cs) => [chat, ...cs]);
    setChatId(chat.id);
    setMessages([]);
    setError(null);
  }

  async function removeChat(id: string) {
    if (!confirm("Delete this chat?")) return;
    await deleteChat(id);
    setChats((cs) => cs.filter((c) => c.id !== id));
    if (id === chatId) {
      setChatId(null);
      setMessages([]);
    }
  }

  // Update the last message (the assistant answer being streamed).
  const updateLast = (f: (m: Message) => Message) =>
    setMessages((ms) => [...ms.slice(0, -1), f(ms[ms.length - 1])]);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const content = input.trim();
    if (!content || streaming) return;

    let id = chatId;
    if (!id) {
      const chat = await createChat();
      id = chat.id;
      setChatId(id);
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
    try {
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
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(String(e));
    } finally {
      setStreaming(false);
      refreshChats(); // new title / ordering
    }
  }

  return (
    <div className="app">
      <aside>
        <button onClick={newChat}>+ New chat</button>
        <ul>
          {chats.map((c) => (
            <li key={c.id} className={c.id === chatId ? "active" : ""}>
              <span onClick={() => openChat(c.id)}>{c.title ?? "New chat"}</span>
              <button className="delete" onClick={() => removeChat(c.id)} title="Delete">
                ×
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main>
        <div className="messages">
          {messages.length === 0 && <p className="empty">Ask a question about hep-th papers.</p>}
          {messages.map((m, i) => (
            <div key={i} className={`message ${m.role}`}>
              <div className="content">
                {m.content ? (
                  <Markdown text={m.content} />
                ) : (
                  streaming && i === messages.length - 1 && "…"
                )}
              </div>
              {m.sources && m.sources.length > 0 && (
                <details className="sources">
                  <summary>{m.sources.length} sources</summary>
                  <ol>
                    {m.sources.map((s) => (
                      <li key={s.n}>
                        {s.url ? (
                          <a href={s.url} target="_blank" rel="noreferrer">
                            <Markdown text={s.title} inline />
                          </a>
                        ) : (
                          <Markdown text={s.title} inline />
                        )}{" "}
                        <small>
                          {s.authors} (arXiv:{s.paper_id}), <Markdown text={s.section} inline />
                        </small>
                        <div className="snippet">{s.snippet}</div>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
            </div>
          ))}
          {error && <div className="error">{error}</div>}
          <div ref={bottom} />
        </div>

        <form onSubmit={send}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) send(e);
            }}
            placeholder="Ask something… LaTeX welcome, e.g. $\chi_i\chi_j$ (Enter to send, Shift+Enter for a new line)"
            rows={3}
          />
          {streaming ? (
            <button type="button" onClick={() => abort.current?.abort()}>
              Stop
            </button>
          ) : (
            <button type="submit" disabled={!input.trim()}>
              Send
            </button>
          )}
        </form>
      </main>
    </div>
  );
}
