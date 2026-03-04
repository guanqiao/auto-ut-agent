# PyUT Agent VS Code Extension Publish Script (Windows)
# Usage: .\publish.ps1 [patch|minor|major]

param(
    [Parameter(Position=0)]
    [ValidateSet("patch", "minor", "major")]
    [string]$VersionBump = "patch"
)

# Colors
$Green = "`e[32m"
$Yellow = "`e[33m"
$Red = "`e[31m"
$NC = "`e[0m"

Write-Host "${Green}=== PyUT Agent VS Code Extension Publish Script ===${NC}"

# Check if vsce is installed
$vsce = Get-Command vsce -ErrorAction SilentlyContinue
if (-not $vsce) {
    Write-Host "${Yellow}Installing vsce...${NC}"
    npm install -g @vscode/vsce
}

# Check for uncommitted changes
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "${Yellow}Warning: You have uncommitted changes${NC}"
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

# Run tests
Write-Host "${Green}Running tests...${NC}"
npm test
if ($LASTEXITCODE -ne 0) {
    Write-Host "${Red}Tests failed. Aborting publish.${NC}"
    exit 1
}

# Build
Write-Host "${Green}Building extension...${NC}"
npm run package

# Bump version
Write-Host "${Green}Bumping version ($VersionBump)...${NC}"
npm version $VersionBump

# Get new version
$packageJson = Get-Content package.json | ConvertFrom-Json
$newVersion = $packageJson.version
Write-Host "${Green}New version: $newVersion${NC}"

# Package
Write-Host "${Green}Packaging extension...${NC}"
vsce package

# Create git tag
Write-Host "${Green}Creating git tag v$newVersion...${NC}"
git tag "v$newVersion"

# Publish
Write-Host "${Green}Publishing to VS Code Marketplace...${NC}"
vsce publish

Write-Host "${Green}=== Publish Complete! ===${NC}"
Write-Host "Version: $newVersion"
Write-Host "Package: pyutagent-vscode-$newVersion.vsix"
Write-Host "${Yellow}Don't forget to push tags: git push --tags${NC}"
