# SPDX-FileCopyrightText: 2026 Chris Labatt-Simon
# SPDX-License-Identifier: GPL-3.0-or-later
<#
.SYNOPSIS
    Set up Transcriber Studio on Windows, starting from nothing.

.DESCRIPTION
    Finds a Python new enough to run the installer, installs one if there is
    none, then hands over to install.py which does the rest.

    This exists because install.py cannot install the Python it is running on.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Check          # report what is installed, change nothing
    .\install.ps1 -Yes            # no prompts
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Yes,
    [switch]$Minimal,
    [switch]$NoGpu,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$MinPython = [version]'3.10'
$WantPython = '3.13'

function Write-Step($text) { Write-Host "`n$text" -ForegroundColor Cyan }
function Write-Ok($text)   { Write-Host "  [ok] $text" -ForegroundColor Green }
function Write-Bad($text)  { Write-Host "  [x] $text"  -ForegroundColor Red }

Write-Host 'Transcriber Studio - Windows setup'

# --- find a Python we can use -------------------------------------------
function Find-Python {
    # The launcher knows about every install; fall back to whatever is on PATH.
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $candidates += (& py -0p 2>$null | ForEach-Object {
            ($_ -split '\s{2,}')[-1].Trim()
        } | Where-Object { $_ -like '*python.exe' })
    }
    $candidates += (Get-Command python -ErrorAction SilentlyContinue |
                    Select-Object -ExpandProperty Source)
    foreach ($exe in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
        try {
            $raw = & $exe -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
            if ($raw -and [version]$raw -ge $MinPython) { return @{ Exe = $exe; Version = $raw } }
        } catch { }
    }
    return $null
}

Write-Step 'Python'
$python = Find-Python
if ($python) {
    Write-Ok "Python $($python.Version) at $($python.Exe)"
} else {
    Write-Host "  No Python $MinPython or newer found."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Bad 'winget is not available, so Python cannot be installed automatically.'
        Write-Host "      Install Python $WantPython from https://www.python.org/downloads/"
        Write-Host '      (tick "Add python.exe to PATH"), then run this again.'
        exit 1
    }
    $go = $Yes -or $DryRun
    if (-not $go) {
        $answer = Read-Host "  Install Python $WantPython with winget? [Y/n]"
        $go = ($answer -eq '' -or $answer -match '^[Yy]')
    }
    if (-not $go) { Write-Bad 'Python is required.'; exit 1 }

    $id = "Python.Python.$WantPython"
    Write-Host "      $ winget install --id $id"
    if (-not $DryRun) {
        winget install --id $id --accept-source-agreements --accept-package-agreements --disable-interactivity
        # winget edits the registry, not this session, so pick the change up now.
        $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
        $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
        $env:Path = "$machinePath;$userPath"
        $python = Find-Python
        if (-not $python) {
            Write-Bad 'Python was installed but is not on PATH in this session.'
            Write-Host '      Close this window, open a new one, and run .\install.ps1 again.'
            exit 1
        }
        Write-Ok "Python $($python.Version) installed"
    } else {
        Write-Host '      (dry run - not installed)'
        exit 0
    }
}

# --- hand over ------------------------------------------------------------
Write-Step 'Handing over to install.py'
$arguments = @((Join-Path $PSScriptRoot 'install.py'))
if ($Check)   { $arguments += '--check' }
if ($Yes)     { $arguments += '--yes' }
if ($Minimal) { $arguments += '--minimal' }
if ($NoGpu)   { $arguments += '--no-gpu' }
if ($DryRun)  { $arguments += '--dry-run' }

& $python.Exe @arguments
exit $LASTEXITCODE
