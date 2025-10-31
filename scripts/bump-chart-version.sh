#!/bin/bash
# Bump chart version and update CHANGELOG
# Usage: ./scripts/bump-chart-version.sh <chart-name> <version-type> [description]
#   version-type: major, minor, or patch
#   description: Optional change description

set -e

CHART_NAME=$1
VERSION_TYPE=$2
DESCRIPTION=${3:-"Version bump"}

CHARTS_DIR="helm/charts"
CHART_PATH="$CHARTS_DIR/$CHART_NAME"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage
usage() {
    echo "Usage: $0 <chart-name> <version-type> [description]"
    echo ""
    echo "Arguments:"
    echo "  chart-name    Name of the chart (e.g., data-platform, ml-platform)"
    echo "  version-type  Type of version bump: major, minor, or patch"
    echo "  description   Optional description of changes (default: 'Version bump')"
    echo ""
    echo "Examples:"
    echo "  $0 data-platform patch 'Fix ClickHouse configuration'"
    echo "  $0 ml-platform minor 'Add new MLflow feature'"
    echo "  $0 api-gateway major 'Breaking API changes'"
    echo ""
    echo "Available charts:"
    ls -1 "$CHARTS_DIR" 2>/dev/null | grep -v "^README" || echo "  No charts found"
    exit 1
}

# Validate arguments
if [ -z "$CHART_NAME" ] || [ -z "$VERSION_TYPE" ]; then
    usage
fi

# Validate chart exists
if [ ! -d "$CHART_PATH" ]; then
    echo -e "${RED}❌ ERROR: Chart not found: $CHART_NAME${NC}"
    echo ""
    echo "Available charts:"
    ls -1 "$CHARTS_DIR"
    exit 1
fi

# Validate Chart.yaml exists
if [ ! -f "$CHART_PATH/Chart.yaml" ]; then
    echo -e "${RED}❌ ERROR: Chart.yaml not found in $CHART_PATH${NC}"
    exit 1
fi

# Get current version
CURRENT_VERSION=$(grep "^version:" "$CHART_PATH/Chart.yaml" | awk '{print $2}')

if [ -z "$CURRENT_VERSION" ]; then
    echo -e "${RED}❌ ERROR: Could not extract version from Chart.yaml${NC}"
    exit 1
fi

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}         Chart Version Bump Tool${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Chart:           ${GREEN}$CHART_NAME${NC}"
echo -e "Current Version: ${YELLOW}$CURRENT_VERSION${NC}"
echo -e "Bump Type:       ${GREEN}$VERSION_TYPE${NC}"
echo ""

# Parse current version
IFS='.' read -r -a VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR="${VERSION_PARTS[0]}"
MINOR="${VERSION_PARTS[1]}"
PATCH="${VERSION_PARTS[2]}"

# Calculate new version
case "$VERSION_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo -e "${RED}❌ ERROR: Invalid version type: $VERSION_TYPE${NC}"
        echo "   Must be one of: major, minor, patch"
        exit 1
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo -e "New Version:     ${GREEN}$NEW_VERSION${NC}"
echo ""

# Confirm
read -p "Proceed with version bump? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Update Chart.yaml
echo -e "${BLUE}→${NC} Updating Chart.yaml..."
sed -i "s/^version: .*/version: $NEW_VERSION/" "$CHART_PATH/Chart.yaml"
echo -e "  ${GREEN}✓${NC} Updated version to $NEW_VERSION"

# Update CHANGELOG.md if it exists
if [ -f "$CHART_PATH/CHANGELOG.md" ]; then
    echo -e "${BLUE}→${NC} Updating CHANGELOG.md..."
    
    DATE=$(date +%Y-%m-%d)
    
    # Create new entry
    NEW_ENTRY="
## [$NEW_VERSION] - $DATE

### Changed
- $DESCRIPTION

"
    
    # Insert after line 6 (after header)
    local temp_file=$(mktemp)
    head -n 6 "$CHART_PATH/CHANGELOG.md" > "$temp_file"
    echo "$NEW_ENTRY" >> "$temp_file"
    tail -n +7 "$CHART_PATH/CHANGELOG.md" >> "$temp_file"
    mv "$temp_file" "$CHART_PATH/CHANGELOG.md"
    
    echo -e "  ${GREEN}✓${NC} Added entry to CHANGELOG.md"
else
    echo -e "${YELLOW}⚠${NC}  No CHANGELOG.md found (consider creating one)"
fi

# Validate the updated chart
echo ""
echo -e "${BLUE}→${NC} Validating updated chart..."
if helm lint "$CHART_PATH" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Chart validation passed"
else
    echo -e "  ${RED}✗${NC} Chart validation failed (check with: helm lint $CHART_PATH)"
fi

# Summary
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Version bump complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Summary:"
echo "  Chart:       $CHART_NAME"
echo "  Old Version: $CURRENT_VERSION"
echo "  New Version: $NEW_VERSION"
echo "  Change Type: $VERSION_TYPE"
echo ""
echo "Next steps:"
echo "  1. Review the changes:"
echo "     git diff $CHART_PATH/Chart.yaml"
echo "     git diff $CHART_PATH/CHANGELOG.md"
echo ""
echo "  2. Test the chart:"
echo "     helm template test-release $CHART_PATH"
echo ""
echo "  3. Commit the changes:"
echo "     git add $CHART_PATH/Chart.yaml $CHART_PATH/CHANGELOG.md"
echo "     git commit -m \"Bump $CHART_NAME to $NEW_VERSION\""
echo ""
echo "  4. Create a pull request or use the promotion workflow"
echo ""
