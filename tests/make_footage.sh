#!/bin/zsh
# Makes the footage the tests and benches import, in tests/out/clips, unless it's already there.
#   input.mov  ProRes 4444 with alpha, 832x464: a test pattern inside a moving, soft-edged circle, the
#              kind of keyed clip a trail is often made from.
#   map.mp4    H.264, 1920x1080: smooth, slowly changing gradients, the kind of clip used as a map.
#   small.mov  ProRes 4444 with alpha, 320x180: the input clip, small, for the suite.
#   small_reversed.mov  The same, back to front, as a reversed layer should show it.
# Each is 300 frames at 30 fps. Needs ffmpeg.
set -eu
DIR=${0:A:h}/out/clips
mkdir -p "$DIR"
command -v ffmpeg >/dev/null || { echo "ffmpeg is needed: brew install ffmpeg"; exit 2; }

# The circle's alpha: white inside a radius that wanders with time, black outside, softened.
circle() {
    local w=$1 h=$2
    echo "geq=lum='255*clip((${h}*0.3-hypot(X-${w}*(0.5+0.3*sin(T*0.9)),Y-${h}*(0.5+0.25*cos(T*1.3))))/6,0,1)'"
}
pattern() {
    local w=$1 h=$2 out=$3
    [ -f "$out" ] && return
    ffmpeg -hide_banner -loglevel error -f lavfi -i "testsrc2=s=${w}x${h}:r=30:d=10" \
        -f lavfi -i "color=black:s=${w}x${h}:r=30:d=10,format=gray,$(circle $w $h)" \
        -filter_complex "[0][1]alphamerge,format=yuva444p10le" \
        -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le "$out"
}
pattern 832 464 "$DIR/input.mov"
pattern 320 180 "$DIR/small.mov"
[ -f "$DIR/small_reversed.mov" ] || ffmpeg -hide_banner -loglevel error -i "$DIR/small.mov" -vf reverse \
    -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le "$DIR/small_reversed.mov"
[ -f "$DIR/map.mp4" ] || ffmpeg -hide_banner -loglevel error \
    -f lavfi -i "gradients=s=1920x1080:r=30:d=10:speed=0.02:n=4,gblur=sigma=20" \
    -c:v libx264 -crf 18 -pix_fmt yuv420p "$DIR/map.mp4"
ls -l "$DIR"
