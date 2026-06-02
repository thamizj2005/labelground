# Determine which Python to use
if [ -f /.dockerenv ]; then
    # Inside Docker
    PYTHON="python3"
    PIP="pip3"
    echo "detected: Running inside Docker container"
else
    # On Host
    if [ -d "venv" ]; then
        PYTHON="./venv/bin/python"
        PIP="./venv/bin/pip"
        echo "detected: Using virtual environment in ./venv"
    else
        PYTHON="python3"
        PIP="pip3"
        echo "detected: Running on host system"
    fi
fi

# Install dependencies if needed
echo ""
echo "Checking dependencies..."
if ! $PYTHON -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    $PIP install -r requirements.txt || echo "Note: Automatic installation failed. You may need to run manual setup."
else
    echo "✓ Dependencies ready"
fi

# Check for port conflict
if lsof -i :8000 > /dev/null 2>&1; then
    echo ""
    echo "⚠️  CRITICAL: Port 8000 is already in use!"
    echo "The server might fail to start. Running processes on port 8000:"
    lsof -i :8000
    echo ""
fi

# Create workspace directory
mkdir -p workspace/projects workspace/logs

# Initialize database
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from database.models import init_database
init_database('workspace/meta.db')
" 2>/dev/null && echo "✓ Database checked"

echo ""
echo "============================================================"
echo "Starting the server..."
echo "Access at: http://localhost:8000"
echo "============================================================"
echo ""

# Run the application
$PYTHON run.py
