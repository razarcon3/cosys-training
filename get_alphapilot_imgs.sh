#!/bin/bash

# Set the URL of the zip file
ZIP_URL="https://s3.amazonaws.com/herox-alphapilot/Data_Training.zip"

# Set the name of the output directory
OUTPUT_DIR="alphapilot_imgs"

# Set the name of the downloaded zip file
ZIP_FILE="Data_Training.zip"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download the zip file using wget
wget -c -O "$ZIP_FILE" "$ZIP_URL"

# Check if the download was successful
if [ $? -eq 0 ]; then
    # Unzip the file into the output directory, overwriting existing files
    # and stripping the top-level directory
    unzip -o -j "$ZIP_FILE" -d "$OUTPUT_DIR"

    # Check if unzip was successful
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded and extracted to $OUTPUT_DIR"

        # Optional: Remove the downloaded zip file
        rm "$ZIP_FILE"
    else
        echo "Error: Failed to unzip the file."
        exit 1
    fi
else
    echo "Error: Failed to download the file."
    exit 1
fi

exit 0