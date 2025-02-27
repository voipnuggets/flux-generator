#!/bin/bash

# Create iconset directory
mkdir -p resources/icon.iconset

# Copy PNG files to iconset with correct names
cp resources/icon_16x16.png resources/icon.iconset/icon_16x16.png
cp resources/icon_32x32.png resources/icon.iconset/icon_32x32.png
cp resources/icon_32x32.png resources/icon.iconset/icon_16x16@2x.png
cp resources/icon_64x64.png resources/icon.iconset/icon_32x32@2x.png
cp resources/icon_128x128.png resources/icon.iconset/icon_128x128.png
cp resources/icon_256x256.png resources/icon.iconset/icon_256x256.png
cp resources/icon_256x256.png resources/icon.iconset/icon_128x128@2x.png
cp resources/icon_512x512.png resources/icon.iconset/icon_512x512.png
cp resources/icon_512x512.png resources/icon.iconset/icon_256x256@2x.png
cp resources/icon_1024x1024.png resources/icon.iconset/icon_512x512@2x.png

# Create icns file
iconutil -c icns resources/icon.iconset

# Clean up temporary files
rm -rf resources/icon.iconset
rm resources/icon_*.png

echo "Created resources/icon.icns" 