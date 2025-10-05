"""
Test suite for DriveDiT.
Comprehensive unit and integration tests for all components.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

__version__ = "0.1.0"