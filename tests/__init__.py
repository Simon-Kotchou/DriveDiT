"""DriveDiT tests: capture contract, tooling and reader, plus a torch smoke test until the model package is rebuilt."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
