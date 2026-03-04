#!/bin/bash

# PyUT Agent VS Code Extension Publish Script
# Usage: ./publish.sh [patch|minor|major]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== PyUT Agent VS Code Extension Publish Script ===${NC}"

# Check if vsce is installed
if ! command -v vsce &> /dev/null; then
    echo -e "${YELLOW}Installing vsce...${NC}"
    npm install -g @vscode/vsce
fi

# Get version bump type
VERSION_BUMP=${1:-patch}

# Validate version bump type
if [[ ! "$VERSION_BUMP" =~ ^(patch|minor|major)$ ]]; then
    echo -e "${RED}Invalid version bump type: $VERSION_BUMP${NC}"
    echo "Usage: ./publish.sh [patch|minor|major]"
    exit 1
fi

# Check for uncommitted changes
if [[ -n $(git status --porcelain) ]]; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run tests
echo -e "${GREEN}Running tests...${NC}"
npm test || {
    echo -e "${RED}Tests failed. Aborting publish.${NC}"
    exit 1
}

# Build
echo -e "${GREEN}Building extension...${NC}"
npm run package

# Bump version
echo -e "${GREEN}Bumping version ($VERSION_BUMP)...${NC}"
npm version $VERSION_BUMP

# Get new version
NEW_VERSION=$(node -p "require('./package.json').version")
echo -e "${GREEN}New version: $NEW_VERSION${NC}"

# Package
echo -e "${GREEN}Packaging extension...${NC}"
vsce package

# Create git tag
echo -e "${GREEN}Creating git tag v$NEW_VERSION...${NC}"
git tag "v$NEW_VERSION"

# Publish
echo -e "${GREEN}Publishing to VS Code Marketplace...${NC}"
vsce publish

echo -e "${GREEN}=== Publish Complete! ===${NC}"
echo -e "Version: $NEW_VERSION"
echo -e "Package: pyutagent-vscode-$NEW_VERSION.vsix"
echo -e "${YELLOW}Don't forget to push tags: git push --tags${NC}"
