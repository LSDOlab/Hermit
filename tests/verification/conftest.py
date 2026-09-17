"""Put ``examples/verification`` on the path so a benchmark's ``_geometry`` /
``_harness`` imports resolve when pytest -- rather than the script itself -- is the
one importing it."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "examples" / "verification"))
