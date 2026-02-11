#!/bin/bash
# Terraform validation script with Checkov security scanning
# Usage: ./validate.sh [terraform_directory]

set -e

TF_DIR="${1:-.}"

echo "=== Terraform Validation Pipeline ==="
echo "Directory: $TF_DIR"
echo ""

# Step 1: Format check
echo "Step 1/4: Checking formatting..."
if terraform -chdir="$TF_DIR" fmt -check -recursive; then
    echo "✓ Formatting OK"
else
    echo "✗ Formatting issues found. Run: terraform fmt -recursive"
    exit 1
fi
echo ""

# Step 2: Validate
echo "Step 2/4: Validating configuration..."
if terraform -chdir="$TF_DIR" validate; then
    echo "✓ Validation passed"
else
    echo "✗ Validation failed"
    exit 1
fi
echo ""

# Step 3: Checkov security scan
echo "Step 3/4: Running Checkov security scan..."
if command -v checkov &> /dev/null; then
    checkov -d "$TF_DIR" --framework terraform --quiet --compact \
        --check HIGH,CRITICAL \
        --output cli
    echo "✓ Security scan complete"
else
    echo "⚠ Checkov not installed. Install with: pip install checkov"
    echo "  Skipping security scan..."
fi
echo ""

# Step 4: Plan (dry-run)
echo "Step 4/4: Running terraform plan..."
if terraform -chdir="$TF_DIR" plan -input=false -no-color; then
    echo "✓ Plan succeeded"
else
    echo "✗ Plan failed"
    exit 1
fi

echo ""
echo "=== All validations passed ==="
