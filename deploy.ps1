# PowerShell script to deploy SHAPE to Hugging Face Spaces
# Usage: .\deploy.ps1 -SpaceName "your-username/shape"

param(
    [Parameter(Mandatory=$true)]
    [string]$SpaceName
)

Write-Host "🚀 Starting deployment to Hugging Face Spaces..." -ForegroundColor Green

# Check if git is initialized
if (-not (Test-Path .git)) {
    Write-Host "📦 Initializing Git repository..." -ForegroundColor Yellow
    git init
    git add .
    git commit -m "Initial commit: SHAPE application"
}

# Check if remote exists
$remoteExists = git remote | Select-String -Pattern "origin"
if (-not $remoteExists) {
    Write-Host "🔗 Adding Hugging Face Space as remote..." -ForegroundColor Yellow
    $spaceUrl = "https://huggingface.co/spaces/$SpaceName"
    git remote add origin $spaceUrl
    Write-Host "Remote added: $spaceUrl" -ForegroundColor Green
} else {
    Write-Host "⚠️  Remote 'origin' already exists. Updating..." -ForegroundColor Yellow
    $spaceUrl = "https://huggingface.co/spaces/$SpaceName"
    git remote set-url origin $spaceUrl
}

# Check current branch
$currentBranch = git branch --show-current
if ($null -eq $currentBranch) {
    Write-Host "📝 Creating main branch..." -ForegroundColor Yellow
    git checkout -b main
    $currentBranch = "main"
}

# Push to Hugging Face
Write-Host "📤 Pushing to Hugging Face Spaces..." -ForegroundColor Yellow
Write-Host "Space: $SpaceName" -ForegroundColor Cyan
Write-Host "Branch: $currentBranch" -ForegroundColor Cyan

try {
    git push -u origin $currentBranch
    Write-Host "✅ Successfully pushed to Hugging Face Spaces!" -ForegroundColor Green
    Write-Host "🌐 Your Space will be available at: https://huggingface.co/spaces/$SpaceName" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "📝 Next steps:" -ForegroundColor Yellow
    Write-Host "1. Go to https://huggingface.co/spaces/$SpaceName/settings" -ForegroundColor White
    Write-Host "2. Add OPENAI_API_KEY as a Repository Secret" -ForegroundColor White
    Write-Host "3. Wait for the build to complete (5-10 minutes)" -ForegroundColor White
} catch {
    Write-Host "❌ Error pushing to Hugging Face Spaces" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "1. Make sure you're logged in: huggingface-cli login" -ForegroundColor White
    Write-Host "2. Verify the Space name is correct: $SpaceName" -ForegroundColor White
    Write-Host "3. Check if the Space exists at https://huggingface.co/spaces" -ForegroundColor White
}

