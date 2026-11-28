#!/usr/bin/env bash
# Reproducible TLA+/TLC runner for Poker Arena formal verification specs.
# Pins tla2tools.jar v1.7.4 with SHA-256 verification and runs deterministic TLC
# checks (-workers 1 -fp 0 -lncheck final) in isolated temporary metadirs.
#
# Usage:
#   verification/run_tlc.sh                 # check all specs with their .cfg
#   verification/run_tlc.sh --check-each    # check each invariant individually + deadlock
#   verification/run_tlc.sh path/to/Spec.tla
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="$ROOT/.tools"
JAR="${TLA2TOOLS:-$TOOLS_DIR/tla2tools.jar}"
JAR_URL="https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar"
JAR_SHA="936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"

pick_java() {
  if [ -n "${JAVA:-}" ] && [ -x "$JAVA" ]; then
    echo "$JAVA"; return 0
  fi
  for cand in \
    "$TOOLS_DIR/jre/Contents/Home/bin/java" \
    "$TOOLS_DIR/jre/bin/java" \
    "/opt/homebrew/opt/openjdk/bin/java" \
    "/usr/local/opt/openjdk/bin/java"; do
    if [ -x "$cand" ]; then
      echo "$cand"; return 0
    fi
  done
  if command -v java >/dev/null 2>&1 && java -version >/dev/null 2>&1; then
    command -v java; return 0
  fi
  echo "ERROR: No Java 11+ runtime found. Set JAVA=/path/to/java or install JRE into .tools/jre." >&2
  return 1
}

ensure_jar() {
  if [ ! -f "$JAR" ]; then
    mkdir -p "$(dirname "$JAR")"
    echo "Downloading pinned tla2tools.jar v1.7.4 into $JAR..." >&2
    curl -fsSL "$JAR_URL" -o "$JAR"
  fi
  if command -v shasum >/dev/null 2>&1; then
    echo "$JAR_SHA  $JAR" | shasum -a 256 -c - >/dev/null
  elif command -v sha256sum >/dev/null 2>&1; then
    echo "$JAR_SHA  $JAR" | sha256sum -c - >/dev/null
  fi
}

CHECK_EACH=0
SPECS=()
for arg in "$@"; do
  if [ "$arg" = "--check-each" ]; then
    CHECK_EACH=1
  else
    SPECS+=("$arg")
  fi
done

if [ "${#SPECS[@]}" -eq 0 ]; then
  SPECS=(
    "$ROOT/verification/RunnerCommitRecovery.tla"
    "$ROOT/verification/DuplicateConcurrency.tla"
  )
fi

JAVA_BIN="$(pick_java)"
ensure_jar

run_one() {
  local spec="$1"
  local cfg="$2"
  shift 2
  local meta
  meta="$(mktemp -d "${TMPDIR:-/tmp}/tlc-XXXXXX")"
  set +e
  "$JAVA_BIN" -XX:+UseParallelGC -cp "$JAR" tlc2.TLC \
    -workers 1 -fp 0 -lncheck final -cleanup \
    ${1+"$@"} \
    -metadir "$meta" -config "$cfg" "$spec"
  local rc=$?
  set -e
  rm -rf "$meta"
  return "$rc"
}

check_each_spec() {
  local spec="$1"
  local cfg="${spec%.tla}.cfg"
  local name
  name="$(basename "${spec%.tla}")"
  local tmpdir
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/tlc-each-XXXXXX")"
  cp "$spec" "$tmpdir/${name}.tla"
  local tmp_spec="$tmpdir/${name}.tla"
  local base_cfg="$tmpdir/base.cfg"
  local invs=()
  local in_inv=0
  while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*INVARIANTS[[:space:]]*$ ]]; then
      in_inv=1
      continue
    fi
    if [ "$in_inv" -eq 1 ]; then
      if [[ "$line" =~ ^[[:space:]]*(SPECIFICATION|CONSTANTS|PROPERTIES|INIT|NEXT|SYMMETRY|CONSTRAINT|ACTION_CONSTRAINT) ]]; then
        in_inv=0
        echo "$line" >> "$base_cfg"
      elif [[ "$line" =~ ^[[:space:]]*([A-Za-z0-9_]+)[[:space:]]*$ ]]; then
        invs+=("${BASH_REMATCH[1]}")
      fi
    else
      echo "$line" >> "$base_cfg"
    fi
  done < "$cfg"

  echo "=== TLC per-property: $name ==="
  local any_fail=0
  for inv in "${invs[@]}"; do
    local icfg="$tmpdir/${inv}.cfg"
    cat "$base_cfg" > "$icfg"
    printf "\nINVARIANTS\n    %s\n" "$inv" >> "$icfg"
    local out
    if out="$(run_one "$tmp_spec" "${inv}.cfg" -deadlock 2>&1)"; then
      printf "  PASS  %s\n" "$inv"
    else
      local rc=$?
      if [ "$rc" -eq 12 ] || [ "$rc" -eq 11 ]; then
        printf "  FAIL  %s\n" "$inv"
      else
        printf "  ERR   %s (exit %d)\n" "$inv" "$rc"
        printf "%s\n" "$out" | tail -n 10 >&2
      fi
      any_fail=1
    fi
  done
  local out
  if out="$(run_one "$tmp_spec" "base.cfg" 2>&1)"; then
    printf "  PASS  NoDeadlock\n"
  else
    local rc=$?
    if [ "$rc" -eq 11 ]; then
      printf "  FAIL  NoDeadlock\n"
    else
      printf "  ERR   NoDeadlock (exit %d)\n" "$rc"
      printf "%s\n" "$out" | tail -n 10 >&2
    fi
    any_fail=1
  fi
  rm -rf "$tmpdir"
  return "$any_fail"
}

status=0
for spec in "${SPECS[@]}"; do
  cfg="${spec%.tla}.cfg"
  name="$(basename "${spec%.tla}")"
  if [ "$CHECK_EACH" -eq 1 ]; then
    if ! check_each_spec "$spec"; then
      status=1
    fi
  else
    echo "=== TLC: $name ==="
    if run_one "$spec" "$cfg"; then
      echo "--- $name: PASSED ---"
    else
      rc=$?
      echo "--- $name: FAILED (exit $rc) ---"
      status=1
    fi
  fi
done

exit "$status"
