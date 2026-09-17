"""Scordelis-Lo roof (quadrilateral control)

This is the standard Scordelis-Lo cylindrical roof benchmark, using the same physical
problem and published reference as ``ex_scordelis_lo.py`` but with quadrilateral
cells. A cylinder is developable, so its structured quadrilaterals are planar. It
therefore serves as the control for the warped-quad finding in issue LSDOlab/Hermit#7:
quadrilaterals themselves are not the problem.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ex_scordelis_lo import make_case  # noqa: E402


CASE = make_case("quad")


if __name__ == "__main__":
    from _harness import main

    main(CASE)
