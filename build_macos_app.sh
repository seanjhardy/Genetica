#!/bin/bash

# Build macOS .app bundle for Genetica
# This creates a proper macOS application with icon that appears in the dock

set -e

echo "Building Genetica for macOS..."

# Build the release binary
cargo build --release

# Define paths
APP_NAME="Genetica"
BUNDLE_NAME="${APP_NAME}.app"
CONTENTS_DIR="${BUNDLE_NAME}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
BINARY_NAME="genetica-rust"
ICON_FILE="assets/icons/dna_highlighted.icns"

# Clean up old bundle if it exists
if [ -d "${BUNDLE_NAME}" ]; then
    echo "Removing old bundle..."
    rm -rf "${BUNDLE_NAME}"
fi

# Create bundle structure
echo "Creating bundle structure..."
mkdir -p "${MACOS_DIR}"
mkdir -p "${RESOURCES_DIR}"

# Copy the binary
echo "Copying binary..."
cp "target/release/${BINARY_NAME}" "${MACOS_DIR}/${APP_NAME}"

# Copy the icon
if [ -f "${ICON_FILE}" ]; then
    echo "Copying icon..."
    cp "${ICON_FILE}" "${RESOURCES_DIR}/AppIcon.icns"
else
    echo "Warning: Icon file not found at ${ICON_FILE}"
fi

# Copy assets folder to Resources (the app will look here when bundled)
if [ -d "assets" ]; then
    echo "Copying assets to bundle..."
    cp -r "assets" "${RESOURCES_DIR}/"
fi

# Create a wrapper script that sets the working directory correctly
echo "Creating launcher wrapper..."
cat > "${MACOS_DIR}/${APP_NAME}_wrapper" << 'WRAPPER_EOF'
#!/bin/bash
# Get the directory containing this script (Contents/MacOS)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go up to Contents, then to Resources where assets are
cd "$DIR/../Resources"
# Run the actual binary
exec "$DIR/Genetica"
WRAPPER_EOF

chmod +x "${MACOS_DIR}/${APP_NAME}_wrapper"

# Rename the original binary and use the wrapper as the main executable
mv "${MACOS_DIR}/${APP_NAME}" "${MACOS_DIR}/Genetica_bin"
mv "${MACOS_DIR}/${APP_NAME}_wrapper" "${MACOS_DIR}/${APP_NAME}"

# Update the wrapper to call the renamed binary
cat > "${MACOS_DIR}/${APP_NAME}" << 'WRAPPER_EOF'
#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../Resources"
exec "$DIR/Genetica_bin"
WRAPPER_EOF

chmod +x "${MACOS_DIR}/${APP_NAME}"

# Create Info.plist
echo "Creating Info.plist..."
cat > "${CONTENTS_DIR}/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>${APP_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.genetica.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF

# Create PkgInfo
echo "Creating PkgInfo..."
echo -n "APPL????" > "${CONTENTS_DIR}/PkgInfo"

echo ""
echo "✅ Bundle created successfully: ${BUNDLE_NAME}"
echo ""
echo "To run the app:"
echo "  open ${BUNDLE_NAME}"
echo ""
echo "To install to Applications:"
echo "  cp -r ${BUNDLE_NAME} /Applications/"
echo ""
