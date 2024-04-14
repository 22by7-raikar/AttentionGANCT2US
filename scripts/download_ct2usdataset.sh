#!/bin/bash

# Check if gdown and unzip are installed
if ! command -v gdown &> /dev/null || ! command -v unzip &> /dev/null; then
    echo "gdown or unzip is not installed. Please install them to proceed."
    exit 1
fi

# Check if a Google Drive link is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 google_drive_link"
    exit 1
fi

# Get the file ID from the Google Drive link
FILE_ID=$(echo "$1" | grep -o -P '(?<=/d/|id=)[^/&\?]+')

if [ -z "$FILE_ID" ]; then
    echo "Invalid Google Drive link."
    exit 1
fi

# Download the file
echo "Downloading file from Google Drive..."
gdown --id "$FILE_ID" || { echo "Failed to download the file."; exit 1; }

# Extract the file
FILE_NAME=$(basename "$1")
unzip -q "$FILE_NAME" || { echo "Failed to extract the file."; exit 1; }

echo "File downloaded and extracted successfully."
