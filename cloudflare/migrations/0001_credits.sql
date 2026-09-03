-- Credit pool ledger (see docs/PRIVACY.md). Nullifiers are hashes of serials:
-- the table cannot be used to learn who bought what, only that a token was spent.
CREATE TABLE IF NOT EXISTS spent (
  nullifier TEXT PRIMARY KEY,
  kid TEXT NOT NULL,
  ts INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS stats (
  kid TEXT PRIMARY KEY,
  issued INTEGER NOT NULL DEFAULT 0,
  redeemed INTEGER NOT NULL DEFAULT 0
);
