#!/usr/bin/env python3
"""Strip Quarto div wrappers and redundant captions from GFM output."""

import re
import sys
from pathlib import Path


def clean_gfm(text: str) -> str:
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Strip <div ...> and </div> lines
        if re.match(r"\s*</?div[\s>]", line):
            i += 1
            continue
        # Strip execution_count attribute lines (continuation of <div>)
        if re.match(r"\s*execution_count=", line):
            i += 1
            continue
        # Strip Figure N: / Table N: caption blocks (may span multiple lines)
        if re.match(r"^(Figure|Table)\s+\d+:", line):
            i += 1
            while i < len(lines) and lines[i].strip():
                # Stop if we hit table or image content
                if lines[i].startswith("|") or lines[i].startswith("!"):
                    break
                i += 1
            continue
        out.append(line)
        i += 1

    text = "\n".join(out)

    # Rewrite local image paths to GitHub raw URLs
    text = re.sub(
        r"!\[([^\]]*)\]\(metr-stats_files/",
        r"![\1](https://raw.githubusercontent.com/JonasMoss/metr-stats/main/metr-stats_files/",
        text,
    )

    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


if __name__ == "__main__":
    path = Path(sys.argv[1])
    text = path.read_text()
    path.write_text(clean_gfm(text))
