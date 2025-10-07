# Clean up repository - remove temporary and test files

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "🧹 Cleaning Up Repository" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$filesToRemove = @(
    # Test scripts
    "simple_test.py",
    "test_search_quality.py",
    "verify_search_works.py",
    "test_auto_labeling.py",
    "example_auto_labeling.py",
    "test_clip_alignment.py",
    "diagnose_scores.py",
    "check_storage.py",
    
    # Temporary processing scripts
    "re_embed_clip_only.py",
    "re_embed_clip_only_parallel.py",
    "re_embed_clip_only_sequential.py",
    
    # Dataset setup scripts (keep only main ones)
    "create_test_subset.py",
    "setup_test_dataset.py",
    "download_test_videos.py",
    "download_test_videos.ps1",
    
    # Download scripts (models already cached)
    "download_clip_model.py",
    "download_blip_model.py",
    "download_ucf101.py",
    
    # Old/backup files
    "convert_avi_to_mp4.py"
)

$removed = 0
$notFound = 0

Write-Host "Removing temporary files..." -ForegroundColor Yellow
Write-Host ""

foreach ($file in $filesToRemove) {
    if (Test-Path $file) {
        Remove-Item $file -Force
        Write-Host "  ✅ Removed: $file" -ForegroundColor Green
        $removed++
    } else {
        Write-Host "  ⏭️  Not found: $file" -ForegroundColor Gray
        $notFound++
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✅ Cleanup Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Summary:"
Write-Host "   Removed: $removed files"
Write-Host "   Not found: $notFound files"
Write-Host ""

Write-Host "📁 Keeping essential files:" -ForegroundColor Yellow
Write-Host "   ✅ ultra_fast_pipeline.py - Main processing"
Write-Host "   ✅ multi_gpu_pipeline.py - Multi-GPU processing"
Write-Host "   ✅ run_pipeline.py - Original pipeline"
Write-Host "   ✅ run_api.py - API server"
Write-Host "   ✅ check_dataset.py - Dataset verification"
Write-Host "   ✅ convert_ucf101_videos.py - Video conversion"
Write-Host "   ✅ process_subset.py - Subset creation"
Write-Host "   ✅ All documentation (*.md)"
Write-Host "   ✅ All source code (src/)"
Write-Host "   ✅ All deployment files (docker, k8s)"
Write-Host ""

Write-Host "🎯 Repository is now clean and production-ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Review remaining files"
Write-Host "  2. Commit changes: git add . && git commit -m 'Clean up repository'"
Write-Host "  3. Push to GitHub: git push origin main"
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
