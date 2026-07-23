"""修复 KiCad 原理图中硬编码的 lib_id 路径为相对路径。

用法:
    python fix_sch.py <path_to.kicad_sch> [lib_name]
    python fix_sch.py output/dw01_sch/schematic.kicad_sch battery_protection

默认将绝对路径中的自定义库名替换为相对路径。
"""
import re
import sys
from pathlib import Path


def fix_lib_paths(filepath: str | Path, lib_name: str = "battery_protection") -> bool:
    """Replace absolute lib_id paths with relative ones in a .kicad_sch file."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: file not found: {filepath}")
        return False

    content = filepath.read_text(encoding="utf-8")

    # Match patterns like: C:\Users\...\battery_protection:XXX → battery_protection:XXX
    old_pattern = re.compile(r'[A-Za-z]:[^"]*' + re.escape(lib_name) + r'[:"]')
    fixed = old_pattern.sub(lib_name + ':', content)

    if fixed == content:
        print(f"No hardcoded paths found for '{lib_name}' in {filepath}")
        return True

    filepath.write_text(fixed, encoding="utf-8")
    print(f"Fixed lib_id paths: {filepath} ({len(old_pattern.findall(content))} occurrences)")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "battery_protection"
    ok = fix_lib_paths(path, lib)
    sys.exit(0 if ok else 1)
