# PowerShell script to deploy to existing Hugging Face Space: 5xuekun/SHAPE
# Usage: .\deploy_to_existing_space.ps1

$SpaceName = "5xuekun/SHAPE"
$SpaceUrl = "https://huggingface.co/spaces/$SpaceName"

Write-Host "🚀 Deploying to existing Space: $SpaceName" -ForegroundColor Green
Write-Host "Space URL: $SpaceUrl" -ForegroundColor Cyan
Write-Host ""

# Check if git is initialized
if (-not (Test-Path .git)) {
    Write-Host "📦 Initializing Git repository..." -ForegroundColor Yellow
    git init
}

# Check current branch
$currentBranch = git branch --show-current 2>$null
if ($null -eq $currentBranch) {
    Write-Host "📝 Creating main branch..." -ForegroundColor Yellow
    git checkout -b main
    $currentBranch = "main"
} elseif ($currentBranch -ne "main") {
    Write-Host "⚠️  Current branch is '$currentBranch', switching to 'main'..." -ForegroundColor Yellow
    git checkout -b main 2>$null
    if ($LASTEXITCODE -ne 0) {
        git checkout main
    }
    $currentBranch = "main"
}

# Check if remote exists
$remoteExists = git remote | Select-String -Pattern "origin"
if ($remoteExists) {
    Write-Host "🔄 Updating existing remote 'origin'..." -ForegroundColor Yellow
    git remote set-url origin $SpaceUrl
} else {
    Write-Host "🔗 Adding remote 'origin'..." -ForegroundColor Yellow
    git remote add origin $SpaceUrl
}

Write-Host "✅ Remote configured: $SpaceUrl" -ForegroundColor Green
Write-Host ""

# Stage all changes
Write-Host "📝 Staging all changes..." -ForegroundColor Yellow
git add .

# Check if there are changes to commit
$status = git status --porcelain
if ($status) {
    Write-Host "💾 Committing changes..." -ForegroundColor Yellow
    $commitMessage = "Update: SHAPE application with analysis visualizer and optimizations"
    git commit -m $commitMessage
} else {
    Write-Host "ℹ️  No changes to commit" -ForegroundColor Cyan
}

# Push to Hugging Face Space
Write-Host ""
Write-Host "📤 Pushing to Hugging Face Space..." -ForegroundColor Yellow
Write-Host "This will replace the existing content in the Space." -ForegroundColor Yellow
Write-Host ""

$response = Read-Host "Continue? (Y/N)"
if ($response -ne "Y" -and $response -ne "y") {
    Write-Host "❌ Deployment cancelled" -ForegroundColor Red
    exit
}

try {
    Write-Host "Pushing to origin/$currentBranch..." -ForegroundColor Cyan
    git push -u origin $currentBranch --force
    
    Write-Host ""
    Write-Host "✅ Successfully deployed to Hugging Face Space!" -ForegroundColor Green
    Write-Host "🌐 Your Space: $SpaceUrl" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📝 Next steps:" -ForegroundColor Yellow
    Write-Host "1. Wait for the build to complete (5-10 minutes)" -ForegroundColor White
    Write-Host "2. Check the Space at: $SpaceUrl" -ForegroundColor White
    Write-Host "3. Verify OPENAI_API_KEY is set in Space Settings > Repository secrets" -ForegroundColor White
    Write-Host "4. Check logs if there are any issues" -ForegroundColor White
} catch {
    Write-Host ""
    Write-Host "❌ Error pushing to Hugging Face Space" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Make sure you're logged in: huggingface-cli login" -ForegroundColor White
    Write-Host "2. Verify you have write access to the Space: $SpaceName" -ForegroundColor White
    Write-Host "3. Try manual push: git push -u origin $currentBranch --force" -ForegroundColor White
}

