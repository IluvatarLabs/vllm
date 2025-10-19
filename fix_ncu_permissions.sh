#!/bin/bash
#
# Fix NCU Permissions - Enable NVIDIA GPU Performance Counter Access
#
# NCU requires special permissions to access GPU performance counters.
# This script enables those permissions.
#

set -e

echo "=========================================="
echo "Fixing NCU Permissions"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "✓ Running as root"
else
    echo "⚠ Not running as root. You may need sudo for some operations."
fi

echo ""
echo "Enabling GPU performance counter access..."
echo ""

# Method 1: Set profiling mode to unrestricted (temporary, lost on reboot)
echo "Method 1: Temporary fix (until reboot)"
echo "-----------------------------------------"
if [ -f /proc/driver/nvidia/params ]; then
    echo "Setting NVreg_RestrictProfilingToAdminUsers=0..."
    if sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" > /etc/modprobe.d/nvidia-profiling.conf'; then
        echo "✓ Modprobe config updated"
        echo ""
        echo "Reloading NVIDIA kernel module..."
        if sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia; then
            echo "✓ NVIDIA module reloaded"
        else
            echo "⚠ Could not reload module. You may need to reboot."
        fi
    else
        echo "✗ Failed to update modprobe config"
    fi
else
    echo "⚠ NVIDIA driver not found at /proc/driver/nvidia/params"
fi

echo ""
echo "Method 2: Immediate fix (current session only)"
echo "-----------------------------------------"
if [ -f /sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers ]; then
    echo "Current value:"
    cat /sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers
    echo ""

    echo "Note: Cannot modify this sysfs parameter directly."
    echo "The modprobe configuration above will take effect after module reload or reboot."
else
    echo "⚠ Parameter file not found"
fi

echo ""
echo "Method 3: Using nvidia-modprobe (if available)"
echo "-----------------------------------------"
if command -v nvidia-modprobe &> /dev/null; then
    echo "Running nvidia-modprobe..."
    sudo nvidia-modprobe || true
    echo "✓ Done"
else
    echo "⚠ nvidia-modprobe not found"
fi

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

# Test NCU access
if command -v ncu &> /dev/null; then
    echo "Testing NCU access with a simple command..."
    if ncu --query-metrics 2>&1 | grep -q "dram__bytes"; then
        echo "✓ NCU can access performance counters!"
    else
        echo "⚠ NCU may still have permission issues"
        echo ""
        echo "Output from ncu --query-metrics:"
        ncu --query-metrics 2>&1 | head -20
    fi
else
    echo "⚠ ncu command not found"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. If the temporary fix worked, you can now run NCU profiling:"
echo "   ./run_ncu_bandwidth_test.sh"
echo ""
echo "2. To make the fix permanent across reboots:"
echo "   - The modprobe config has been created at:"
echo "     /etc/modprobe.d/nvidia-profiling.conf"
echo "   - It will be loaded on next boot"
echo ""
echo "3. If you still see permission errors, you may need to:"
echo "   - Reboot the system for changes to take effect"
echo "   - OR run the profiling command with sudo:"
echo "     sudo ./run_ncu_bandwidth_test.sh"
echo ""
echo "4. Alternative: Run the microbench directly with sudo:"
echo "   sudo python3 tools/profiling/run_nwor_microbench.py \\"
echo "     --scenario short --requests 8 --batches 2 --draft-tokens 4 \\"
echo "     --temperature 0.7 --nwor-modes off --scv-modes off \\"
echo "     --enable-ncu --ncu-metrics \"dram__bytes_write.sum\" \\"
echo "     --output test_ncu.json"
echo ""

# Show current NVIDIA driver version
echo "Current NVIDIA Driver Info:"
echo "----------------------------"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

echo "Done!"
