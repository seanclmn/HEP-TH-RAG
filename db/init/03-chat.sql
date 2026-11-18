-- Chat history for the API (api/). Idempotent: the API also applies this on startup,
-- because docker-entrypoint-initdb.d scripts only run on a fresh volume.

CREATE TABLE IF NOT EXISTS chats (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    title       text,                                 -- first question, if not given
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()    -- bumped on every message
);

CREATE TABLE IF NOT EXISTS messages (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chat_id     uuid NOT NULL REFERENCES chats (id) ON DELETE CASCADE,
    role        text NOT NULL CHECK (role IN ('user', 'assistant')),
    content     text NOT NULL,
    sources     jsonb,      -- assistant only: the retrieved chunks shown to the LLM
    model       text,       -- assistant only: the Ollama model that wrote it
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_chat_id_idx ON messages (chat_id, id);
CREATE INDEX IF NOT EXISTS chats_updated_at_idx ON chats (updated_at DESC);
