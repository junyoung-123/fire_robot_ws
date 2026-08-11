param(
    [int]$Retries = 30,
    [int]$SleepMs = 1000
)

Add-Type -AssemblyName System.Windows.Forms

Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public static class Win32WindowTools {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", CharSet=CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll")]
    public static extern bool MoveWindow(IntPtr hWnd, int X, int Y, int nWidth, int nHeight, bool bRepaint);

    [DllImport("user32.dll")]
    public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
}
"@

function Get-TopLevelWindows {
    $items = New-Object System.Collections.Generic.List[object]
    $callback = [Win32WindowTools+EnumWindowsProc]{
        param([IntPtr]$hWnd, [IntPtr]$lParam)
        if ([Win32WindowTools]::IsWindowVisible($hWnd)) {
            $sb = New-Object System.Text.StringBuilder 512
            [void][Win32WindowTools]::GetWindowText($hWnd, $sb, $sb.Capacity)
            $title = $sb.ToString()
            if (-not [string]::IsNullOrWhiteSpace($title)) {
                $items.Add([pscustomobject]@{
                    Handle = $hWnd
                    Title = $title
                })
            }
        }
        return $true
    }
    [void][Win32WindowTools]::EnumWindows($callback, [IntPtr]::Zero)
    return $items
}

function Find-WindowByTitle {
    param([string[]]$Patterns)
    foreach ($window in Get-TopLevelWindows) {
        foreach ($pattern in $Patterns) {
            if ($window.Title -match $pattern) {
                return $window
            }
        }
    }
    return $null
}

function Find-WindowsByTitle {
    param([string[]]$Patterns)
    $foundWindows = New-Object System.Collections.Generic.List[object]
    foreach ($window in Get-TopLevelWindows) {
        foreach ($pattern in $Patterns) {
            if ($window.Title -match $pattern) {
                $foundWindows.Add($window)
                break
            }
        }
    }
    return $foundWindows
}

$screen = [System.Windows.Forms.Screen]::PrimaryScreen.WorkingArea
$leftWidth = [Math]::Floor($screen.Width * 0.50)
$rightWidth = $screen.Width - $leftWidth
$rightTopRatio = 0.64

for ($attempt = 1; $attempt -le $Retries; $attempt++) {
    $gazebo = Find-WindowByTitle @('Gazebo', 'gzclient')
    $rviz = Find-WindowByTitle @('RViz', 'rviz2')
    $cameraWindows = @(Find-WindowsByTitle @('Front Camera', 'Front.*Ubuntu', 'rqt_image_view', 'Image View', '/camera/color/image_raw'))
    $hasCamera = $cameraWindows.Count -gt 0

    if ($gazebo -ne $null) {
        [void][Win32WindowTools]::ShowWindow($gazebo.Handle, 9)
        [void][Win32WindowTools]::MoveWindow(
            $gazebo.Handle,
            $screen.Left,
            $screen.Top,
            $leftWidth,
            $screen.Height,
            $true)
    }

    if ($rviz -ne $null) {
        $rvizTop = $screen.Top
        $rvizHeight = $screen.Height
        if ($hasCamera) {
            $rvizHeight = [Math]::Floor($screen.Height * $rightTopRatio)
        }
        [void][Win32WindowTools]::ShowWindow($rviz.Handle, 9)
        [void][Win32WindowTools]::MoveWindow(
            $rviz.Handle,
            $screen.Left + $leftWidth,
            $rvizTop,
            $rightWidth,
            $rvizHeight,
            $true)
    }

    foreach ($camera in $cameraWindows) {
        $cameraTop = $screen.Top + [Math]::Floor($screen.Height * $rightTopRatio)
        $cameraHeight = $screen.Height - [Math]::Floor($screen.Height * $rightTopRatio)
        [void][Win32WindowTools]::ShowWindow($camera.Handle, 9)
        [void][Win32WindowTools]::MoveWindow(
            $camera.Handle,
            $screen.Left + $leftWidth,
            $cameraTop,
            $rightWidth,
            $cameraHeight,
            $true)
    }

    if ($gazebo -ne $null -and $rviz -ne $null -and $hasCamera) {
        Write-Host "Arranged Gazebo, RViz, and camera windows."
        exit 0
    }

    if ($gazebo -ne $null -and $rviz -ne $null -and $attempt -eq $Retries) {
        Write-Host "Arranged Gazebo and RViz windows. Camera window was not found."
        exit 0
    }

    Start-Sleep -Milliseconds $SleepMs
}

Write-Warning "Could not find Gazebo/RViz windows. Available windows:"
Get-TopLevelWindows | Select-Object -ExpandProperty Title | Sort-Object | Write-Host
exit 1
