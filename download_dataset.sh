#!/bin/bash

# Create the target directory
mkdir -p Million_AID
cd Million_AID

# Download all parts of the split zip file
echo "Downloading dataset parts..."
for i in $(seq -f "%03g" 1 17)
do
    wget "https://huggingface.co/datasets/torchgeo/million-aid/resolve/main/test/test.zip.${i}?download=true" -O "test.zip.${i}"
done

# Combine all parts into a single zip file
echo "Combining parts into a single zip file..."
cat test.zip.* > full_test.zip

# Unzip the combined file into Million_AID folder
echo "Extracting files..."
unzip full_test.zip

# Cleanup: remove the split zip parts and the combined zip file
echo "Cleaning up..."
rm test.zip.*
rm full_test.zip

echo "Dataset is ready in Million_AID directory."
