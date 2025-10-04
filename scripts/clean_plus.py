from pathlib import Path
p = Path(r'C:\Users\kvaru\Downloads\rag_video_search_pipeline\src\core\vector_search_client.py')
text = p.read_text(encoding='utf-8')
lines = text.splitlines()
new_lines = [ln for ln in lines if not ln.lstrip().startswith('+')]
if len(new_lines) == len(lines):
    print('No leading + lines found')
else:
    p.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
    print(f'Removed {len(lines)-len(new_lines)} lines starting with +')
