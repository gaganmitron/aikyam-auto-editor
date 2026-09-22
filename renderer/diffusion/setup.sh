#!/bin/sh
# Fetch diffusionstudio/editor (MPL-2.0) at a PINNED commit into vendor/ (unmodified, not committed) and install only what the headless harness needs.
set -e
SHA=0add88db090cd865bda7050af4a5979bc9482b58
cd "$(dirname "$0")"
[ -d vendor/editor/.git ] || git clone --filter=blob:none https://github.com/diffusionstudio/editor.git vendor/editor
git -C vendor/editor fetch -q origin "$SHA" 2>/dev/null || true
git -C vendor/editor checkout -q "$SHA"
cd vendor/editor
npm install --ignore-scripts --no-audit --no-fund \
  --workspace packages/assets --workspace packages/jsx --workspace packages/koota-solid --workspace packages/runtime --workspace packages/reconciler --workspace packages/encoder
npx --yes patch-package          # the project patches koota at install time; without it entities cannot gain traits
cd ../..
npm install --no-audit --no-fund
