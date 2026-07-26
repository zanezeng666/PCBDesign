"""ERC 探测：验证"网络标签直接放在引脚端点"能否实现电气连通。

用 KiCad 自带 python 运行（也可用普通 python，纯文本生成）：
  python scripts/_erc_probe.py
然后调用 kicad-cli sch erc 检查输出。
"""
import subprocess
import sys
import uuid
from pathlib import Path


def uid():
    return str(uuid.uuid4())


RES_SYM = """    (symbol "probe:R"
      (pin_names (offset 0)) (exclude_from_sim no) (in_bom yes) (on_board yes)
      (property "Reference" "R" (at 0 2.54 0) (effects (font (size 1.27 1.27))))
      (property "Value" "R" (at 0 -2.54 0) (effects (font (size 1.27 1.27))))
      (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
      (symbol "R_0_1"
        (rectangle (start -1.016 2.54) (end 1.016 -2.54)
          (stroke (width 0.254) (type default)) (fill (type none)))
        (pin passive line (at 0 5.08 270) (length 2.54)
          (name "1" (effects (font (size 1.27 1.27))))
          (number "1" (effects (font (size 1.27 1.27)))))
        (pin passive line (at 0 -5.08 90) (length 2.54)
          (name "2" (effects (font (size 1.27 1.27))))
          (number "2" (effects (font (size 1.27 1.27))))))
    )"""


def place(ref, x, y):
    return f"""  (symbol (lib_id "probe:R") (at {x} {y} 0) (unit 1)
    (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)
    (uuid {uid()})
    (property "Reference" "{ref}" (at {x} {y-3} 0) (effects (font (size 1.27 1.27))))
    (property "Value" "10k" (at {x} {y+3} 0) (effects (font (size 1.27 1.27))))
    (property "Footprint" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
    (property "Datasheet" "" (at 0 0 0) (effects (font (size 1.27 1.27)) hide))
    (instances
      (project "" (path "/{uid()}" (reference "{ref}") (unit 1))))
    (pin "1" (uuid {uid()}))
    (pin "2" (uuid {uid()}))
  )"""


def label(text, x, y):
    return f"""  (label "{text}" (at {x} {y} 0) (fields_autoplaced yes)
    (effects (font (size 1.27 1.27)) (justify left))
    (uuid {uid()})
  )"""


def build():
    # R1 at (50,50): pin1 endpoint (50, 44.92), pin2 endpoint (50, 55.08)
    # R2 at (80,50): pin1 endpoint (80, 44.92), pin2 endpoint (80, 55.08)
    lines = []
    lines.append("(kicad_sch")
    lines.append("  (version 20250114)")
    lines.append('  (generator "probe")')
    lines.append('  (generator_version "10.0")')
    lines.append(f"  (uuid {uid()})")
    lines.append('  (paper "A4")')
    lines.append("  (lib_symbols")
    lines.append(RES_SYM)
    lines.append("  )")
    lines.append(place("R1", 50, 50))
    lines.append(place("R2", 80, 50))
    # 标签直接放在引脚端点：R1.2 与 R2.1 同名 N1 → 应连通
    lines.append(label("N1", 50, 55.08))   # R1 pin2 endpoint
    lines.append(label("N1", 80, 44.92))   # R2 pin1 endpoint
    lines.append(label("VCC", 50, 44.92))  # R1 pin1
    lines.append(label("GND", 80, 55.08))  # R2 pin2
    lines.append(")")
    return "\n".join(lines)


def main():
    out = Path("work/debug/erc_probe")
    out.mkdir(parents=True, exist_ok=True)
    sch = out / "probe.kicad_sch"
    sch.write_text(build(), encoding="utf-8")
    print("wrote", sch)
    cli = r"E:\KiCad\bin\kicad-cli.exe"
    rpt = out / "erc.rpt"
    r = subprocess.run(
        [cli, "sch", "erc", "--exit-code-violations", "-o", str(rpt), str(sch)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print("returncode:", r.returncode)
    print("stdout:", r.stdout)
    print("stderr:", r.stderr)
    if rpt.exists():
        print("--- report ---")
        print(rpt.read_text(encoding="utf-8", errors="replace"))


if __name__ == "__main__":
    main()
