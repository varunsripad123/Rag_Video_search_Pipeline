"""Process a subset of UCF101 for quick testing."""

import shutil
from pathlib import Path

print("=" * 70)
print("📋 Creating Test Subset from UCF101")
print("=" * 70)

source_dir = Path("ground_clips_mp4")
subset_dir = Path("ground_clips_mp4_subset")

# Select 20 diverse actions, 10 videos each = 200 videos
test_actions = [
    "ApplyEyeMakeup",
    "Archery",
    "Basketball",
    "Biking",
    "Bowling",
    "Diving",
    "Drumming",
    "GolfSwing",
    "HorseRiding",
    "JumpRope",
    "Kayaking",
    "PlayingGuitar",
    "PushUps",
    "RockClimbingIndoor",
    "Skiing",
    "SoccerJuggling",
    "Surfing",
    "TennisSwing",
    "WalkingWithDog",
    "YoYo"
]

videos_per_action = 10
total_copied = 0

# Remove old subset if exists
if subset_dir.exists():
    shutil.rmtree(subset_dir)

subset_dir.mkdir(exist_ok=True)

print(f"\n🎯 Selecting {len(test_actions)} diverse actions")
print(f"   {videos_per_action} videos per action = {len(test_actions) * videos_per_action} total\n")

for action in test_actions:
    action_source = source_dir / action
    action_dest = subset_dir / action
    
    if not action_source.exists():
        print(f"  ⏭️  Skip: {action} (not found)")
        continue
    
    videos = list(action_source.glob("*.mp4"))
    
    if not videos:
        print(f"  ⏭️  Skip: {action} (no videos)")
        continue
    
    action_dest.mkdir(exist_ok=True)
    
    copied = 0
    for video in videos[:videos_per_action]:
        dest_file = action_dest / video.name
        shutil.copy(video, dest_file)
        copied += 1
    
    total_copied += copied
    print(f"  ✅ {action:20s} - copied {copied} videos")

print(f"\n📊 Total: {total_copied} videos in test subset")
print(f"📂 Location: {subset_dir.absolute()}")

print("\n" + "=" * 70)
print("✅ Test Subset Ready!")
print("=" * 70)

print("\n🔄 Quick test workflow:")
print("\n1. Backup main dataset:")
print("   move ground_clips_mp4 ground_clips_mp4_full")
print("")
print("2. Use test subset:")
print("   move ground_clips_mp4_subset ground_clips_mp4")
print("")
print("3. Process subset (~10-15 minutes):")
print("   python run_pipeline.py --enable-labeling")
print("")
print("4. Test search:")
print("   python run_api.py")
print("   # Open: http://localhost:8081/static/index.html")
print("")
print("5. If working well, restore full dataset:")
print("   move ground_clips_mp4 ground_clips_mp4_subset")
print("   move ground_clips_mp4_full ground_clips_mp4")
print("   python run_pipeline.py --enable-labeling  # Process all 13K videos")
print("")
print("=" * 70)
