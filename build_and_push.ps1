# Chạy trong PowerShell trên Windows
$ErrorActionPreference = "Stop"
$RepoDir = "camhub-ai"
$GitHubUrl = "https://github.com/RaymondRMT/camhub-ai.git" # Đổi thành link repo của bạn

# Nếu đang đứng sẵn bên trong thư mục dự án thì không tạo lồng thêm thư mục con
if (-not (Test-Path ".\main.py")) {
    New-Item -ItemType Directory -Force -Path "$RepoDir\static" | Out-Null
    New-Item -ItemType Directory -Force -Path "$RepoDir\android-launcher" | Out-Null
    Set-Location $RepoDir
}

if (-not (Test-Path ".\.git")) {
    git init
}
git branch -M main
git add .
git commit -m "Release CamHub AI Pro v1.1 - Termux Edge Camera Server" --allow-empty

$existingRemotes = git remote
if ($existingRemotes -contains "origin") {
    git remote set-url origin $GitHubUrl
} else {
    git remote add origin $GitHubUrl
}

git push -u origin main
Write-Host "Hoan tat! Toan bo ma nguon da duoc day len GitHub." -ForegroundColor Cyan
