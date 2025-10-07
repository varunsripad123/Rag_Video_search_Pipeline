"""Check if dataset is properly organized."""

from pathlib import Path

print("=" * 70)
print("📊 Dataset Check")
print("=" * 70)

root = Path("ground_clips_mp4")

if not root.exists():
    print(f"\n❌ Directory not found: {root.absolute()}")
    print("\nCreate it with:")
    print(f"  mkdir {root}")
    exit(1)

# Get all subdirectories
folders = [d for d in root.iterdir() if d.is_dir()]

if not folders:
    print(f"\n❌ No folders found in {root}")
    print("\nAdd your video folders like:")
    print(f"  {root}/walking/")
    print(f"  {root}/running/")
    exit(1)

print(f"\n📁 Found {len(folders)} action categories:\n")

total_videos = 0
total_avi = 0
for folder in sorted(folders):
    mp4_videos = list(folder.glob("*.mp4"))
    avi_videos = list(folder.glob("*.avi"))
    
    total_videos += len(mp4_videos)
    total_avi += len(avi_videos)
    
    if len(mp4_videos) > 0:
        status = "✅"
        count_str = f"{len(mp4_videos)} mp4"
    elif len(avi_videos) > 0:
        status = "🔄"
        count_str = f"{len(avi_videos)} avi (need conversion)"
    else:
        status = "⚠️"
        count_str = "0 videos"
    
    print(f"  {status} {folder.name:20s} - {count_str}")

print(f"\n📊 Total: {total_videos} mp4 videos")
if total_avi > 0:
    print(f"      + {total_avi} avi videos (need conversion)")

if total_videos == 0 and total_avi == 0:
    print("\n❌ No video files found!")
    print("\nMake sure your videos are:")
    print("  • In .mp4 or .avi format")
    print("  • Inside action folders")
    print(f"  • Located in: {root.absolute()}")
elif total_videos == 0 and total_avi > 0:
    print("\n🔄 Found .avi videos that need conversion!")
    print("\nConvert them to .mp4 format:")
    print("  python convert_ucf101_videos.py")
elif len(folders) == 1:
    print("\n⚠️  Only 1 action category detected!")
    print("   Add more diverse actions for better search results.")
elif total_videos < 10:
    print("\n⚠️  Only a few videos detected.")
    print("   Add more videos for better testing.")
else:
    print("\n✅ Dataset looks good!")
    print("\n🔄 Next steps:")
    print("   1. python re_embed_clip_only.py")
    print("   2. python run_api.py")
    print("   3. Open: http://localhost:8081/static/index.html")

print("=" * 70)
