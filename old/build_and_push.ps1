# Chạy trong PowerShell trên Windows
$RepoDir = "camhub-ai"
$GitHubUrl = "https://github.com/your-username/camhub-ai.git" # Đổi thành link repo của bạn

New-Item -ItemType Directory -Force -Path "$RepoDir\static" | Out-Null
New-Item -ItemType Directory -Force -Path "$RepoDir\android-launcher" | Out-Null

# Di chuyển hoặc lưu 8 tệp trên vào đúng vị trí trong $RepoDir, sau đó chạy lệnh Git:
Set-Location $RepoDir
git init
git branch -M main
git add .
git commit -m "Release CamHub AI Pro v1.0 - Termux Edge Camera Server"
git remote add origin $GitHubUrl
git push -u origin main

Write-Host "Hoan tat! Toan bo ma nguon da duoc day len GitHub." -ForegroundColor Cyan