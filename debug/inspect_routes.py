import sys
import os

from app.mcp.asgi import app

for route in app.routes:
    # Most routes have .path
    path = getattr(route, "path", None)
    print(f"Route: {path} - {type(route)}")
