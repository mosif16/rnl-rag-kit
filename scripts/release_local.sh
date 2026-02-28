#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="false"
SKIP_BUILD="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --skip-build)
      SKIP_BUILD="true"
      shift
      ;;
    -h|--help)
      cat <<USAGE
Usage: scripts/release_local.sh [--dry-run] [--skip-build]
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] $*"
  else
    eval "$@"
  fi
}

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required" >&2
  exit 1
fi

if [[ "$SKIP_BUILD" != "true" ]]; then
  run "swift build -c release"
  run "swift test -c release"
fi

run "git fetch --tags origin"

latest_tag="$(git tag -l 'v*' --sort=-version:refname | head -n 1)"
if [[ -z "$latest_tag" ]]; then
  next_tag="v0.1.0"
else
  current="${latest_tag#v}"
  if [[ ! "$current" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Latest tag is not SemVer: $latest_tag" >&2
    exit 1
  fi

  IFS='.' read -r major minor patch <<< "$current"
  patch=$((patch + 1))
  next_tag="v${major}.${minor}.${patch}"
fi

if git rev-parse "$next_tag" >/dev/null 2>&1; then
  echo "Tag already exists: $next_tag" >&2
  exit 1
fi

run "git tag -a '$next_tag' -m 'Release $next_tag'"
run "git push origin '$next_tag'"
run "gh release create '$next_tag' --generate-notes --verify-tag"

echo "Created release: $next_tag"
