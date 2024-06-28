function Test-Administrator {
    $currentUser = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Warning "관리자 권한이 필요합니다. 관리자 권한으로 다시 실행 중..."
    Start-Process powershell "-File `"$PSCommandPath`"" -Verb RunAs
    exit
}

$ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$ffmpeg_zip = "ffmpeg-release-essentials.zip"
$ffmpeg_folder = "C:\ffmpeg"

# FFmpeg 다운로드
Invoke-WebRequest -Uri $ffmpeg_url -OutFile $ffmpeg_zip

# 압축 해제 (-Force 옵션 사용)
Expand-Archive -Path $ffmpeg_zip -DestinationPath $ffmpeg_folder -Force

# FFmpeg의 정확한 경로 설정
$ffmpeg_bin = "$ffmpeg_folder\ffmpeg-7.0.1-essentials_build\bin"

# 환경 변수 추가
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine)
if ($env:Path -notcontains $ffmpeg_bin) {
    [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$ffmpeg_bin", [System.EnvironmentVariableTarget]::Machine)
}

Write-Host "FFmpeg 설치가 완료되었습니다."
