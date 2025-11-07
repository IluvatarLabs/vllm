#!/bin/bash
# Install dependencies for semantic similarity evaluation

echo "Installing evaluation dependencies..."
echo ""

pip install bert-score sentence-transformers scipy transformers torch numpy

echo ""
echo "Checking for AMX-capable CPU..."
if lscpu | grep -q "amx"; then
    echo "✓ AMX support detected!"
    echo ""
    echo "For best AMX performance, also install Intel Extension for PyTorch:"
    echo "  pip install intel_extension_for_pytorch"
    echo ""
    read -p "Install Intel Extension for PyTorch now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install intel_extension_for_pytorch
    fi
else
    echo "ℹ AMX not detected (standard CPU optimizations will be used)"
fi

echo ""
echo "Installation complete!"
echo ""
echo "To run the evaluation:"
echo "  python3 evaluate_semantic_similarity.py"
