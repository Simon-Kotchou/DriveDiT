#!/bin/bash
# Build script for DriveDiT Rust data loading library
#
# This script builds the Rust extension using maturin and installs it
# into the current Python environment.
#
# Prerequisites:
#   - Rust toolchain (rustup install stable)
#   - Python 3.8+ with pip
#   - maturin (pip install maturin)
#
# Usage:
#   ./scripts/build_rust.sh         # Build and install in release mode
#   ./scripts/build_rust.sh dev     # Build and install in development mode
#   ./scripts/build_rust.sh clean   # Clean build artifacts
#   ./scripts/build_rust.sh test    # Run Rust tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
RUST_DIR="$PROJECT_ROOT/rust/drivedit-data"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  DriveDiT Rust Data Loader Builder${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check dependencies
check_dependencies() {
    echo -e "\n${YELLOW}Checking dependencies...${NC}"

    # Check Rust
    if ! command -v cargo &> /dev/null; then
        echo -e "${RED}Error: Rust/Cargo not found${NC}"
        echo "Please install Rust: https://rustup.rs/"
        exit 1
    fi
    echo -e "  ${GREEN}â${NC} Rust $(rustc --version | cut -d' ' -f2)"

    # Check Python
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        echo -e "${RED}Error: Python not found${NC}"
        exit 1
    fi
    PYTHON=$(command -v python3 || command -v python)
    echo -e "  ${GREEN}â${NC} Python $($PYTHON --version | cut -d' ' -f2)"

    # Check pip
    if ! $PYTHON -m pip --version &> /dev/null; then
        echo -e "${RED}Error: pip not found${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}â${NC} pip $($PYTHON -m pip --version | cut -d' ' -f2)"

    # Check/install maturin
    if ! $PYTHON -c "import maturin" &> /dev/null; then
        echo -e "  ${YELLOW}Installing maturin...${NC}"
        $PYTHON -m pip install maturin
    fi
    echo -e "  ${GREEN}â${NC} maturin $($PYTHON -c 'import maturin; print(maturin.__version__)' 2>/dev/null || echo 'installed')"
}

# Function to clean build artifacts
clean_build() {
    echo -e "\n${YELLOW}Cleaning build artifacts...${NC}"
    cd "$RUST_DIR"

    if [ -d "target" ]; then
        rm -rf target
        echo -e "  ${GREEN}â${NC} Removed target/"
    fi

    # Remove Python build artifacts
    find "$PROJECT_ROOT" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type f -name "*.so" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -type f -name "*.pyd" -delete 2>/dev/null || true

    echo -e "  ${GREEN}â${NC} Cleaned Python artifacts"
}

# Function to run Rust tests
run_tests() {
    echo -e "\n${YELLOW}Running Rust tests...${NC}"
    cd "$RUST_DIR"

    cargo test --all-features

    echo -e "\n${GREEN}All tests passed!${NC}"
}

# Function to build in development mode
build_dev() {
    echo -e "\n${YELLOW}Building in development mode...${NC}"
    cd "$RUST_DIR"

    # Build with maturin in development mode
    maturin develop

    echo -e "\n${GREEN}Development build complete!${NC}"
}

# Function to build release
build_release() {
    echo -e "\n${YELLOW}Building release version...${NC}"
    cd "$RUST_DIR"

    # Build with maturin in release mode
    maturin build --release

    # Install the wheel
    WHEEL=$(ls -t target/wheels/*.whl 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        echo -e "  Installing wheel: $(basename $WHEEL)"
        $PYTHON -m pip install --force-reinstall "$WHEEL"
    fi

    echo -e "\n${GREEN}Release build complete!${NC}"
}

# Function to verify installation
verify_installation() {
    echo -e "\n${YELLOW}Verifying installation...${NC}"

    $PYTHON -c "
import sys
try:
    import drivedit_data
    print(f'  â drivedit_data version: {drivedit_data.version()}')
    print(f'  â Rust backend available: {drivedit_data.is_available()}')
    print(f'  â CPU cores detected: {drivedit_data.num_cpus()}')
    sys.exit(0)
except ImportError as e:
    print(f'  â Import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'  â Error: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        echo -e "\n${GREEN}Installation verified successfully!${NC}"
    else
        echo -e "\n${RED}Installation verification failed${NC}"
        exit 1
    fi
}

# Function to run benchmarks
run_benchmarks() {
    echo -e "\n${YELLOW}Running benchmarks...${NC}"
    cd "$RUST_DIR"

    cargo bench

    echo -e "\n${GREEN}Benchmarks complete!${NC}"
}

# Function to show help
show_help() {
    echo "
Usage: $0 [command]

Commands:
    (none)      Build and install in release mode (default)
    dev         Build and install in development mode (faster, with debug symbols)
    clean       Clean all build artifacts
    test        Run Rust unit tests
    bench       Run Rust benchmarks
    verify      Verify installation
    help        Show this help message

Examples:
    $0              # Build release and install
    $0 dev          # Build debug and install
    $0 clean test   # Clean and run tests
"
}

# Main logic
main() {
    cd "$PROJECT_ROOT"

    # Check if Rust directory exists
    if [ ! -d "$RUST_DIR" ]; then
        echo -e "${RED}Error: Rust project not found at $RUST_DIR${NC}"
        exit 1
    fi

    # Parse arguments
    if [ $# -eq 0 ]; then
        # Default: build release
        check_dependencies
        build_release
        verify_installation
    else
        for cmd in "$@"; do
            case "$cmd" in
                dev)
                    check_dependencies
                    build_dev
                    verify_installation
                    ;;
                clean)
                    clean_build
                    ;;
                test)
                    check_dependencies
                    run_tests
                    ;;
                bench)
                    check_dependencies
                    run_benchmarks
                    ;;
                verify)
                    verify_installation
                    ;;
                help|--help|-h)
                    show_help
                    ;;
                *)
                    echo -e "${RED}Unknown command: $cmd${NC}"
                    show_help
                    exit 1
                    ;;
            esac
        done
    fi
}

# Run main
main "$@"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}  Build script completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
