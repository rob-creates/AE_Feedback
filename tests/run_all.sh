#!/bin/zsh
# Runs the whole suite: builds, rebuilds the test projects and the live session in After Effects,
# renders everything with aerender, and checks it.  run_all.sh LABEL [BASELINE]
# Needs After Effects closed, and a build configured with -DFEEDBACK_LOG=ON. With a BASELINE label,
# also lists the frames that moved most since that run. Exits non-zero if the build or any check fails.
set -u
LABEL=$1
BASE=${2:-}
APP="Adobe After Effects 2027"
HERE=${0:A:h}
ROOT=${HERE:h}
LOG=~/Library/Logs/AEFeedback.log

pgrep -x "After Effects" >/dev/null && { echo "After Effects is open: close it first"; exit 3; }
grep -q "FEEDBACK_LOG:BOOL=ON" "$ROOT/Build/CMakeCache.txt" || { echo "Configure with -DFEEDBACK_LOG=ON first"; exit 3; }
if ! built=$(cmake --build "$ROOT/Build" -j8 2>&1); then
    echo "$built" | grep -E "error" | head -20
    echo "The build failed, so nothing was tested"
    exit 2
fi

cd "$HERE"
: > $LOG
rm -rf out/live "out/$LABEL"
mkdir -p out/live

# Each script in a fresh After Effects, which is then quit.
open -a "$APP"
# Scripting works once a script can write a file. After Effects's reply can't be relied on: it can come
# back as 0 even when the script ran.
marker="$HERE/out/.scripting_ready"
rm -f "$marker"
for i in {1..60}; do
    sleep 5
    osascript -e "with timeout of 5 seconds" -e "tell application \"$APP\" to DoScript \"var f = new File('$marker'); f.open('w'); f.write('ready'); f.close();\"" -e "end timeout" >/dev/null 2>&1
    [ -f "$marker" ] && break
done
[ -f "$marker" ] || { echo "After Effects can't run scripts: quit and reopen it, then try again"; exit 4; }
rm -f "$marker"
for script in build_project.jsx live_session.jsx; do
    reply=$(osascript -e "with timeout of 900 seconds" -e "tell application \"$APP\" to DoScriptFile \"$HERE/$script\"" -e "end timeout" 2>&1)
    echo "$script: exit $? ${reply:0:200}"
done
osascript -e "with timeout of 60 seconds" -e "tell application \"$APP\" to DoScript \"app.quit()\"" -e "end timeout" >/dev/null
for i in {1..24}; do pgrep -x "After Effects" >/dev/null || break; sleep 5; done
# It can hang on the way out with no window left, once the projects are saved, so then it's stopped.
if pgrep -x "After Effects" >/dev/null; then
    echo "After Effects hung quitting, so it was stopped"
    pkill -x "After Effects"
    for i in {1..6}; do pgrep -x "After Effects" >/dev/null || break; sleep 5; done
    pkill -9 -x "After Effects"
    sleep 2
fi
pgrep -x "After Effects" >/dev/null && { echo "After Effects would not quit"; exit 4; }

failed=0
report=$(mktemp)
caffeinate -i python3 run_tests.py render "$LABEL" > "$report" 2>&1 || failed=1
grep -E "exit [1-9]|Error|Traceback" "$report"
python3 run_tests.py check "$LABEL" > "$report" 2>&1 || failed=1
grep -E "^FAIL|failed|passed|Traceback|Error" "$report"
python3 live_check.py "$LABEL" > "$report" 2>&1 || failed=1
grep -E "^FAIL|failed|passed|Traceback|Error" "$report"
rm -f "$report"
errors=$(grep -E '^frame=.*err=[1-9]' $LOG | grep -vc 'err=517')
echo "render errors: $errors, CPU renders: $(grep -c 'CPU render' $LOG)"
[ "$errors" -eq 0 ] || failed=1

if [ -n "$BASE" ]; then
    python3 - "$BASE" "$LABEL" <<'PY'
import sys
import numpy as np
from run_tests import read_tiff, OUT
a, b = sys.argv[1:3]
worst, n = [], 0
for folder in sorted((OUT / b).iterdir()):
    if not folder.is_dir() or not (OUT / a / folder.name).is_dir():
        continue
    peak, where = 0, ""
    for t in folder.glob("f_*.tif"):
        other = OUT / a / folder.name / t.name
        if other.exists():
            n += 1
            d = int(np.abs(read_tiff(t) - read_tiff(other)).max())
            if d > peak:
                peak, where = d, t.name
    worst.append((peak, folder.name, where))
worst.sort(reverse=True)
print(f"{a} against {b}: {n} frames, largest differences {worst[:6]}")
PY
fi
rm -rf __pycache__

# A run takes about 10 GB, so only the two newest are kept, and never this run or its baseline. A run
# is a folder in out/ with a timing.json.
keep=0
for timing in $(ls -t out/*/timing.json 2>/dev/null); do
    run=${timing:h:t}
    keep=$((keep + 1))
    if [ $keep -gt 2 ] && [ "$run" != "$LABEL" ] && [ "$run" != "$BASE" ]; then
        rm -rf "out/$run"
        echo "removed the old run $run"
    fi
done
exit $failed
