"""Quick test to see if API server can start without errors."""

try:
    print("Testing API server import...")
    from src.api.server import build_app
    print("✅ Import successful!")
    
    print("\nBuilding app...")
    app = build_app()
    print("✅ App built successfully!")
    
    print("\nAPI server is ready to run!")
    print("Start it with: python run_api.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
