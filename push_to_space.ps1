# Quick script to push to existing Space: 5xuekun/SHAPE
# This will force push and replace existing content

$SpaceName = "5xuekun/SHAPE"

Write-Host "🚀 Pushing to existing Space: $SpaceName" -ForegroundColor Green
Write-Host ""

# Check if we're on main branch
$currentBranch = git branch --show-current
if ($currentBranch -ne "main") {
    Write-Host "⚠️  Current branch is '$currentBranch', switching to 'main'..." -ForegroundColor Yellow
    git checkout -b main 2>$null
    if ($LASTEXITCODE -ne 0) {
        git checkout main
    }
}

# Add all changes
Write-Host "📝 Staging all changes..." -ForegroundColor Yellow
git add .

# Commit
Write-Host "💾 Committing changes..." -ForegroundColor Yellow
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
git commit -m "Update: SHAPE with analysis visualizer and optimizations - $timestamp"

# Force push
Write-Host ""
Write-Host "📤 Force pushing to $SpaceName..." -ForegroundColor Yellow
Write-Host "⚠️  This will replace all existing content in the Space!" -ForegroundColor Red
Write-Host ""

git push -u origin main --force

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Successfully pushed to $SpaceName!" -ForegroundColor Green
    Write-Host "🌐 Space URL: https://huggingface.co/spaces/$SpaceName" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⏳ Wait 5-10 minutes for the build to complete" -ForegroundColor Yellow
} else {
    Write-Host ""
    Write-Host "❌ Push failed. Check the error message above." -ForegroundColor Red
}

