import py_compile
import pathlib
import sys
import traceback

root = pathlib.Path(r"c:\Users\kvaru\Downloads\rag_video_search_pipeline")
failed = []
for p in root.rglob('*.py'):
    try:
        py_compile.compile(str(p), doraise=True)
    except Exception:
        print(f"FAILED: {p}")
        traceback.print_exc()
        failed.append(p)

if failed:
    print(f"\nTotal failures: {len(failed)}")
    sys.exit(1)

print('All .py files compiled successfully')
