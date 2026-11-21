#!/bin/zsh
# Builds Feedback for distribution to Apple silicon Macs: a release build without logging, signed with
# a Developer ID, notarized by Apple and stapled, in a disk image that is signed, notarized and stapled
# too, so Gatekeeper accepts it even offline.
#
#   scripts/package.sh [NOTARY_PROFILE]
#
# Signing needs a "Developer ID Application" certificate in the keychain. Notarizing needs a notarytool
# keychain profile, made once in a terminal (it asks for an app-specific password from appleid.apple.com):
#
#   xcrun notarytool store-credentials NOTARY_PROFILE --apple-id YOUR_APPLE_ID --team-id YOUR_TEAM_ID
#
# Without a profile it stops once the plugin is signed. Writes Build-Release/ and dist/.
set -euo pipefail

ROOT=${0:A:h:h}
BUILD="$ROOT/Build-Release"
DIST="$ROOT/dist"
PROFILE=${1:-}

IDENTITY=$(security find-identity -v -p codesigning | grep -o '"Developer ID Application: [^"]*"' | head -1 | tr -d '"')
[[ -n "$IDENTITY" ]] || { echo "No Developer ID Application certificate in the keychain" >&2; exit 1; }

# The test build in Build/ logs every render, so the release has a build of its own.
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DFEEDBACK_LOG=OFF >/dev/null
cmake --build "$BUILD" -j8 >/dev/null
PLUGIN="$BUILD/Feedback.plugin"
VERSION=$(/usr/libexec/PlistBuddy -c "Print CFBundleShortVersionString" "$PLUGIN/Contents/Info.plist")
[[ "$(lipo -archs "$PLUGIN/Contents/MacOS/Feedback")" == "arm64" ]] || { echo "Not an arm64 build" >&2; exit 1; }

# The build signs ad hoc; distribution needs the Developer ID, a secure timestamp and the hardened runtime.
codesign --force --timestamp --options runtime --sign "$IDENTITY" "$PLUGIN"
codesign --verify --strict --verbose=2 "$PLUGIN"
echo "Signed Feedback $VERSION with $IDENTITY"

if [[ -z "$PROFILE" ]]; then
    echo "No notary profile given, so not notarized. See the top of this script."
    exit 0
fi

notarize() {
    echo "Notarizing $(basename "$1")..."
    xcrun notarytool submit "$1" --keychain-profile "$PROFILE" --wait --timeout 1h
}

rm -rf "$DIST"
mkdir -p "$DIST/Feedback $VERSION"

# The plugin first, so its own ticket travels with it once it's copied out of the disk image.
ditto -c -k --keepParent "$PLUGIN" "$DIST/Feedback.zip"
notarize "$DIST/Feedback.zip"
rm "$DIST/Feedback.zip"
xcrun stapler staple "$PLUGIN"
ditto "$PLUGIN" "$DIST/Feedback $VERSION/Feedback.plugin"
cat > "$DIST/Feedback $VERSION/Install.txt" <<EOF
Feedback $VERSION for Adobe After Effects, for Macs with Apple silicon (macOS 13 or later).

Quit After Effects, copy Feedback.plugin into
/Applications/Adobe After Effects <version>/Plug-ins/
(an admin password may be asked for), then start After Effects. Feedback is under Effect > Stylize.
EOF

DMG="$DIST/Feedback-$VERSION-arm64.dmg"
hdiutil create -quiet -volname "Feedback $VERSION" -srcfolder "$DIST/Feedback $VERSION" -fs HFS+ -format UDZO "$DMG"
codesign --force --timestamp --sign "$IDENTITY" "$DMG"
notarize "$DMG"
xcrun stapler staple "$DMG"
rm -rf "$DIST/Feedback $VERSION"

spctl --assess --type open --context context:primary-signature --verbose=2 "$DMG"
echo "Ready to distribute: $DMG"
