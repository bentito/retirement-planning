#!/bin/bash
mkdir -p stlite_lib
MANIFEST_URL="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.63.1/build/asset-manifest.json"
BASE_URL="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.63.1/build/"

echo "Fetching manifest..."
curl -k -s "$MANIFEST_URL" > manifest.json

# Extract filenames (values from the 'files' object)
# jq -r '.files | values[]' will get "./stlite.js", "./static/js/..."
FILES=$(jq -r '.files | values[]' manifest.json)

for FILE in $FILES; do
    # Remove leading ./
    CLEAN_FILE="${FILE#./}"
    
    # Construct URLs
    SRC_URL="${BASE_URL}${CLEAN_FILE}"
    DEST_FILE="stlite_lib/${CLEAN_FILE}"
    
    # Create directory
    mkdir -p "$(dirname "$DEST_FILE")"
    
    echo "Downloading $CLEAN_FILE..."
    curl -k -s -L -o "$DEST_FILE" "$SRC_URL"
done

echo "Download complete."
rm manifest.json