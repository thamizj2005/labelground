#!/bin/bash
# Install python-multipart in Docker container

echo "Installing python-multipart..."

# Try conda first (has proper permissions in your Docker)
if command -v conda &> /dev/null; then
    echo "Using conda..."
    conda install -y python-multipart -c conda-forge
else
    echo "Using pip..."
    pip install --user python-multipart
fi

echo "✓ Installation complete!"
echo "Now restart the server by pressing Ctrl+C and running ./start.sh again"
