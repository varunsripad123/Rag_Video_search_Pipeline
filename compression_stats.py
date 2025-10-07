"""Calculate compression statistics."""
import json

data = json.load(open('data/processed/metadata.json'))

# Calculate totals
total_encoded = sum(e['byte_size'] for e in data)
total_original = sum(e['byte_size'] * e['ratio'] for e in data)
avg_ratio = sum(e['ratio'] for e in data) / len(data)

saved = total_original - total_encoded
saved_percent = (saved / total_original) * 100

print("\n" + "="*60)
print("📊 VIDEO COMPRESSION & STORAGE ANALYSIS")
print("="*60)
print(f"\n📹 Videos Processed: {len(data)} chunks")
print(f"\n💾 Original Size:    {total_original/1024/1024:>8.2f} MB")
print(f"🗜️  Compressed Size:  {total_encoded/1024/1024:>8.2f} MB")
print(f"💰 Storage Saved:    {saved/1024/1024:>8.2f} MB")
print(f"\n📉 Average Compression: {avg_ratio:.2f}x")
print(f"✅ Storage Reduction:   {saved_percent:.1f}%")
print(f"\n🎯 You're using only {100-saved_percent:.1f}% of original storage!")
print("="*60 + "\n")
