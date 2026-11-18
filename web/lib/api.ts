// Client for the FastAPI backend (api/main.py).

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Source = {
  n: number;
  paper_id: string;
  title: string;
  authors: string;
  section: string;
  chunk_index: number;
  url: string | null;
  score: number;
  snippet: string;
};

export type Chat = { id: string; title: string | null; created_at: string; updated_at: string };

export type Message = {
  id: number | null; // null while an answer is still streaming
  role: "user" | "assistant";
  content: string;
  sources: Source[] | null;
};

export type ChatDetail = Chat & { messages: Message[] };

/** fetch, with a readable error when the API server is down or blocks the origin (CORS). */
async function apiFetch(path: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(`${API_URL}${path}`, init);
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new Error(
      `Cannot reach the API at ${API_URL}. Is it running (.venv/bin/uvicorn api.main:app)? ` +
        `If it is, check that CORS_ORIGINS includes ${window.location.origin}.`,
    );
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await apiFetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const listChats = () => request<Chat[]>("/chats");
export const getChat = (id: string) => request<ChatDetail>(`/chats/${id}`);
export const createChat = () => request<Chat>("/chats", { method: "POST", body: "{}" });
export const deleteChat = (id: string) => request<void>(`/chats/${id}`, { method: "DELETE" });

export type StreamHandlers = {
  onSources: (sources: Source[]) => void;
  onDelta: (text: string) => void;
  onError: (detail: string) => void;
  onDone: (messageId: number | null) => void;
};

/** POST a question and dispatch the Server-Sent Events of the streamed answer. */
export async function sendMessage(
  chatId: string,
  content: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await apiFetch(`/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`${res.status} ${await res.text()}`);

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      dispatch(buffer.slice(0, end), handlers);
      buffer = buffer.slice(end + 2);
    }
  }
}

function dispatch(block: string, h: StreamHandlers) {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (!data.length) return;
  const payload = JSON.parse(data.join("\n"));
  if (event === "sources") h.onSources(payload);
  else if (event === "delta") h.onDelta(payload.text);
  else if (event === "error") h.onError(payload.detail);
  else if (event === "done") h.onDone(payload.message_id);
}
