"""Convert UCF101 .avi videos to .mp4 format."""

import subprocess
import sys
from pathlib import Path

print("=" * 70)
print("🎬 Converting UCF101 Videos: .avi → .mp4")
print("=" * 70)

# Check if ffmpeg is installed
def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

if not check_ffmpeg():
    print("\n❌ ffmpeg not found!")
    print("\n📦 Installing ffmpeg...")
    print("\nOption 1 - Using conda (if you have it):")
    print("  conda install ffmpeg")
    print("\nOption 2 - Download manually:")
    print("  1. Visit: https://www.gyan.dev/ffmpeg/builds/")
    print("  2. Download: ffmpeg-release-essentials.zip")
    print("  3. Extract and add to PATH")
    print("\nOption 3 - Install via pip:")
    print("  pip install ffmpeg-python")
    
    choice = input("\nInstall via pip now? (yes/no): ").strip().lower()
    if choice == "yes":
        subprocess.run([sys.executable, "-m", "pip", "install", "ffmpeg-python"])
        print("\n⚠️  Note: You still need the ffmpeg binary installed.")
        print("   Download from: https://www.gyan.dev/ffmpeg/builds/")
    exit(1)

print("\n✅ ffmpeg found!")

# Source and destination
source_dir = Path("ground_clips_mp4")
if not source_dir.exists():
    print(f"\n❌ Directory not found: {source_dir}")
    exit(1)

# Find all .avi files
avi_files = list(source_dir.rglob("*.avi"))

if not avi_files:
    print(f"\n❌ No .avi files found in {source_dir}")
    print("\nCheck if videos are in a different location.")
    exit(1)

print(f"\n📊 Found {len(avi_files)} .avi files")

# Convert each file
converted = 0
failed = 0
skipped = 0

for avi_file in avi_files:
    mp4_file = avi_file.with_suffix(".mp4")
    
    # Skip if .mp4 already exists
    if mp4_file.exists():
        print(f"⏭️  Skip: {avi_file.name} (already converted)")
        skipped += 1
        continue
    
    print(f"🔄 Converting: {avi_file.name}", end="", flush=True)
    
    try:
        # Convert using ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(avi_file),
                "-c:v", "libx264",      # H.264 video codec
                "-c:a", "aac",           # AAC audio codec
                "-strict", "experimental",
                "-y",                    # Overwrite output
                str(mp4_file)
            ],
            capture_output=True,
            check=True
        )
        
        # Delete original .avi file after successful conversion
        avi_file.unlink()
        
        print(" ✅")
        converted += 1
        
    except subprocess.CalledProcessError as e:
        print(" ❌")
        print(f"   Error: {e.stderr.decode()[:100]}")
        failed += 1

print("\n" + "=" * 70)
print("✅ Conversion Complete!")
print("=" * 70)
print(f"\n📊 Results:")
print(f"   ✅ Converted: {converted} videos")
print(f"   ⏭️  Skipped: {skipped} videos (already .mp4)")
if failed > 0:
    print(f"   ❌ Failed: {failed} videos")

if converted > 0:
    print("\n🔄 Next steps:")
    print("   1. Verify videos: python check_dataset.py")
    print("   2. Re-embed: python re_embed_clip_only.py")
    print("   3. Start API: python run_api.py")

print("=" * 70)
