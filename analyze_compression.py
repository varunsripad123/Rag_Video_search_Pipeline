"""Analyze compression and storage savings."""
import json
from pathlib import Path

# Load manifest
manifest_path = Path("data/processed/metadata.json")
data = json.load(open(manifest_path))

# Calculate stats
total_entries = len(data)
compression_ratios = [e['ratio'] for e in data]
encoded_sizes = [e['byte_size'] for e in data]

avg_ratio = sum(compression_ratios) / len(compression_ratios)
total_encoded = sum(encoded_sizes)
total_original = sum([e['byte_size'] * e['ratio'] for e in data])
storage_saved = total_original - total_encoded
savings_percent = (storage_saved / total_original) * 100

print("="*60)
print("📊 STORAGE COMPRESSION ANALYSIS")
print("="*60)
print(f"\n📹 Total Videos Processed: {total_entries}")
print(f"\n💾 Original Size:  {total_original/1024/1024:.2f} MB")
print(f"🗜️  Encoded Size:   {total_encoded/1024/1024:.2f} MB")
print(f"💰 Storage Saved:  {storage_saved/1024/1024:.2f} MB")
print(f"\n📉 Compression Ratio: {avg_ratio:.2f}x")
print(f"✅ Storage Reduction: {savings_percent:.1f}%")
print(f"\n🎯 That means you're using only {100-savings_percent:.1f}% of original storage!")
print("="*60)

# Show some examples
print("\n📋 Sample Compression Examples:")
for i in range(min(5, len(data))):
    entry = data[i]
    orig = entry['byte_size'] * entry['ratio'] / 1024
    comp = entry['byte_size'] / 1024
    print(f"  {i+1}. {Path(entry['chunk_path']).name[:40]:40} | {orig:6.1f} KB → {comp:6.1f} KB ({entry['ratio']:.1f}x)")
print("="*60)
