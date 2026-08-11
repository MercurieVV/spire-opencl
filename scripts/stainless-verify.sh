#!/usr/bin/env bash
set -euo pipefail

# Stainless (https://github.com/epfl-lara/stainless) is invoked here as a
# standalone CLI, not as a compiler plugin or via sbt-stainless: both of
# those integrations are unreliable across build tools and Scala versions.
# Only verify a small, side-effect-free "pure kernel" of functions listed
# in stainless.conf -- Stainless's supported subset excludes most of the
# Scala/Java standard library.

cd "$(git rev-parse --show-toplevel)"

if ! command -v stainless >/dev/null 2>&1; then
  echo "Stainless CLI not found on PATH."
  echo "Install it from https://github.com/epfl-lara/stainless/releases"
  echo "and make sure the 'stainless' binary is runnable."
  echo "Skipping formal verification for now (not blocking)."
  exit 0
fi

if [ ! -f stainless.conf ]; then
  echo "No stainless.conf found; nothing to verify."
  exit 0
fi

FILES=$(grep -v '^[[:space:]]*#' stainless.conf | grep -v '^[[:space:]]*$' || true)

if [ -z "$FILES" ]; then
  echo "stainless.conf lists no files; nothing to verify."
  exit 0
fi

echo "Running Stainless verification on:"
echo "$FILES"
# shellcheck disable=SC2086
stainless $FILES
