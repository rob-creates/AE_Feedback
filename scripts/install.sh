#!/bin/zsh
# Links Build/Feedback.plugin into After Effects' Plug-ins folder, so each rebuild is picked up
# the next time AE starts. Needs write access to that folder, so run it from an admin account
# or through sudo.
set -euo pipefail

SOURCE="$(cd "$(dirname "$0")/.." && pwd)/Build/Feedback.plugin"
DEST_DIR="/Applications/Adobe After Effects 2027/Plug-ins/Feedback"

if [[ ! -d "$SOURCE" ]]; then
    echo "Not built yet: $SOURCE" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
ln -sfn "$SOURCE" "$DEST_DIR/Feedback.plugin"
echo "Linked $DEST_DIR/Feedback.plugin -> $SOURCE"
echo "Restart After Effects to load it."
