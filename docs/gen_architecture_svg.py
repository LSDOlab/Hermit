#!/usr/bin/env python3
"""Generate the lightweight ShellDomain API overview used by the documentation."""
from pathlib import Path

Path(__file__).parent.joinpath("src", "hermit_architecture.svg").write_text('''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="180" viewBox="0 0 900 180">
<style>text{font-family:sans-serif;fill:#202830}.box{fill:#f4f7fa;stroke:#52667a;stroke-width:2}</style>
<rect width="900" height="180" fill="white"/><rect class="box" x="25" y="55" width="180" height="70" rx="8"/><rect class="box" x="255" y="55" width="180" height="70" rx="8"/><rect class="box" x="485" y="55" width="180" height="70" rx="8"/><rect class="box" x="715" y="55" width="160" height="70" rx="8"/>
<text x="50" y="85">ShellDomain</text><text x="50" y="108">mesh + forms</text><text x="280" y="85">Material, Loads, BCs</text><text x="310" y="108">Fields</text><text x="535" y="85">solve</text><text x="510" y="108">ShellState</text><text x="745" y="85">outputs</text><text x="730" y="108">Field / scalar</text>
<path d="M205 90h50M435 90h50M665 90h50" stroke="#52667a" stroke-width="2"/>
</svg>''')
