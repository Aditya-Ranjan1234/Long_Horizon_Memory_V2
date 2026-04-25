import sys
import os

# Add the ui directory to sys.path so we can import app
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ui"))

try:
    from app import app
except ImportError:
    # Fallback for different directory structures in Vercel
    sys.path.append(os.path.join(os.getcwd(), "ui"))
    from app import app

# This is required for Vercel to pick up the FastAPI instance
app = app
