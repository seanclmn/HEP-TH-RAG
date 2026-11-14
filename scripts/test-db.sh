#!/usr/bin/env bash
# Smoke test for the pgvector database. Usage: ./scripts/test-db.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { echo "Missing .env (copy .env.example)"; exit 1; }
set -a; . ./.env; set +a

psql_db() {
  docker compose exec -T db psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
    -v ON_ERROR_STOP=1 -qtA "$@"
}

pass=0; fail=0
check() { # name, expected, actual
  if [ "$2" = "$3" ]; then echo "PASS  $1"; pass=$((pass+1))
  else echo "FAIL  $1 (expected '$2', got '$3')"; fail=$((fail+1)); fi
}

status=$(docker inspect -f '{{.State.Health.Status}}' hep-th-rag-db 2>/dev/null || echo missing)
check "container healthy" healthy "$status"
[ "$status" = healthy ] || { echo "Start it with: docker compose up -d"; exit 1; }

check "vector extension installed" 1 \
  "$(psql_db -c "SELECT count(*) FROM pg_extension WHERE extname='vector';")"

# Everything below runs in one transaction that is rolled back, so it leaves no data behind.
out=$(psql_db <<'SQL'
BEGIN;
CREATE TEMP TABLE t (id int PRIMARY KEY, embedding vector(3));
INSERT INTO t VALUES (1,'[1,0,0]'), (2,'[0,1,0]'), (3,'[0,0,1]'), (4,'[0.9,0.1,0]');
CREATE INDEX ON t USING hnsw (embedding vector_cosine_ops);
SET LOCAL enable_seqscan = off;
SELECT 'l2:'     || round(('[1,2,3]'::vector <-> '[4,5,6]'::vector)::numeric, 4);
SELECT 'cosine:' || round(('[1,0]'::vector <=> '[0,1]'::vector)::numeric, 4);
SELECT 'knn:'    || string_agg(id::text, ',' ORDER BY d) FROM (
  SELECT id, embedding <=> '[1,0.05,0]' AS d FROM t ORDER BY d LIMIT 2) s;
SELECT 'dims:'   || vector_dims(embedding) FROM t WHERE id = 1;
ROLLBACK;
SQL
)
get() { echo "$out" | grep "^$1:" | cut -d: -f2; }
check "L2 distance [1,2,3]<->[4,5,6]" 5.1962 "$(get l2)"
check "cosine distance of orthogonal vectors" 1.0000 "$(get cosine)"
check "HNSW nearest neighbours of [1,0.05,0]" "1,4" "$(get knn)"
check "vector dimensions" 3 "$(get dims)"

# Dimension mismatch must be rejected.
if psql_db -c "SELECT '[1,2]'::vector(3);" >/dev/null 2>&1; then
  check "rejects wrong dimensions" error ok
else
  check "rejects wrong dimensions" error error
fi

echo "---"; echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ]
