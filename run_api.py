"""Start the API server with web interface."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.server import build_app
from src.config import load_config
import uvicorn

def main():
    """Run the API server."""
    config_path = Path("config/pipeline.yaml")
    config = load_config(config_path)
    
    app = build_app()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║           AI Video Search Platform - API Server              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

🚀 Server starting...

📍 API Endpoint:  http://{config.api.host}:{config.api.port}
🌐 Web Interface: http://localhost:{config.api.port}/static/index.html
📊 Health Check:  http://localhost:{config.api.port}/health
📖 API Docs:      http://localhost:{config.api.port}/docs

🔑 API Key: {config.security.api_keys[0] if config.security.api_keys else 'Not configured'}

Press Ctrl+C to stop the server
""")
    
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level="info",
    )

if __name__ == "__main__":
    main()
