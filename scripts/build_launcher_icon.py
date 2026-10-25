from __future__ import annotations

import struct
import sys
from pathlib import Path


def pack_png_as_ico(source: Path, destination: Path) -> None:
    png = source.read_bytes()
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) < 24:
        raise ValueError(f"Not a valid PNG: {source}")
    width, height = struct.unpack(">II", png[16:24])
    if not (1 <= width <= 256 and 1 <= height <= 256):
        raise ValueError(f"ICO PNG dimensions must be 1..256, got {width}x{height}")

    # ICO embeds PNG data directly for Windows Vista and newer.
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack(
        "<BBBBHHII",
        0 if width == 256 else width,
        0 if height == 256 else height,
        0,
        0,
        1,
        32,
        len(png),
        22,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + entry + png)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/build_launcher_icon.py input.png output.ico")
    pack_png_as_ico(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Wrote {sys.argv[2]}")
