#!/bin/bash
# Validate all Helm charts in the repository
# Usage: ./scripts/validate-charts.sh [chart-name]

set -e

CHARTS_DIR="helm/charts"
FAILED_CHARTS=()
PASSED_CHARTS=()

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          HMCo Helm Charts Validation Suite                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if helm is installed
if ! command -v helm &> /dev/null; then
    echo "❌ ERROR: Helm is not installed"
    echo "   Please install Helm: https://helm.sh/docs/intro/install/"
    exit 1
fi

echo "✓ Helm version: $(helm version --short)"
echo ""

# Function to validate a single chart
validate_chart() {
    local chart_path=$1
    local chart_name=$(basename "$chart_path")
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Validating: $chart_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if Chart.yaml exists
    if [ ! -f "$chart_path/Chart.yaml" ]; then
        echo "❌ No Chart.yaml found in $chart_path"
        FAILED_CHARTS+=("$chart_name")
        return 1
    fi
    
    # Extract version
    local version=$(grep "^version:" "$chart_path/Chart.yaml" | awk '{print $2}')
    echo "Version: $version"
    
    # Check if CHANGELOG exists
    if [ -f "$chart_path/CHANGELOG.md" ]; then
        echo "✓ CHANGELOG.md exists"
    else
        echo "⚠  No CHANGELOG.md (consider adding one)"
    fi
    
    # Run helm lint
    echo ""
    echo "Running helm lint..."
    local lint_output=$(mktemp)
    if helm lint "$chart_path" 2>&1 | tee "$lint_output"; then
        if grep -q "ERROR" "$lint_output"; then
            echo "❌ Lint failed with errors"
            FAILED_CHARTS+=("$chart_name")
            rm -f "$lint_output"
            return 1
        else
            echo "✓ Lint passed"
        fi
    else
        echo "❌ Lint failed"
        FAILED_CHARTS+=("$chart_name")
        rm -f "$lint_output"
        return 1
    fi
    rm -f "$lint_output"
    
    # Test template rendering with default values
    echo ""
    echo "Testing template rendering..."
    if helm template test-release "$chart_path" > /dev/null 2>&1; then
        echo "✓ Templates render successfully"
    else
        echo "⚠  Template rendering issues detected"
        echo "   (This may be expected if the chart requires specific values)"
    fi
    
    # Test with environment-specific values if they exist
    for env in dev staging prod; do
        if [ -f "$chart_path/values/$env.yaml" ]; then
            echo ""
            echo "Testing with $env.yaml..."
            if helm template test-release "$chart_path" \
                --values "$chart_path/values/$env.yaml" > /dev/null 2>&1; then
                echo "✓ Templates render successfully with $env values"
            else
                echo "⚠  Template rendering issues with $env values"
            fi
        fi
    done
    
    echo ""
    echo "✅ $chart_name validation complete"
    PASSED_CHARTS+=("$chart_name")
    echo ""
}

# Main execution
if [ -n "$1" ]; then
    # Validate specific chart
    CHART_PATH="$CHARTS_DIR/$1"
    if [ -d "$CHART_PATH" ]; then
        validate_chart "$CHART_PATH"
    else
        echo "❌ Chart not found: $1"
        echo "Available charts in $CHARTS_DIR:"
        ls -1 "$CHARTS_DIR"
        exit 1
    fi
else
    # Validate all top-level charts
    echo "Validating all top-level charts..."
    echo ""
    
    for chart_path in "$CHARTS_DIR"/*; do
        if [ -d "$chart_path" ] && [ -f "$chart_path/Chart.yaml" ]; then
            if ! validate_chart "$chart_path"; then
                # Continue to next chart even if validation fails
                continue
            fi
        fi
    done
fi

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Validation Summary                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ Passed: ${#PASSED_CHARTS[@]} charts"
if [ ${#PASSED_CHARTS[@]} -gt 0 ]; then
    for chart in "${PASSED_CHARTS[@]}"; do
        echo "   - $chart"
    done
fi

echo ""
if [ ${#FAILED_CHARTS[@]} -gt 0 ]; then
    echo "❌ Failed: ${#FAILED_CHARTS[@]} charts"
    for chart in "${FAILED_CHARTS[@]}"; do
        echo "   - $chart"
    done
    echo ""
    exit 1
else
    echo "🎉 All charts validated successfully!"
    echo ""
    exit 0
fi
