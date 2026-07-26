"""分析 F.Cu SVG，判断 zone 铜皮是否铺到 U1.6 焊盘附近。"""
import re
from pathlib import Path

svg = Path("data/ic_templates/DW01-G/fcu.svg").read_text(encoding="utf-8")
print("len=", len(svg))
m = re.search(r'viewBox="([^"]+)"', svg)
print("viewBox=", m.group(1) if m else None)
print("paths=", svg.count("<path"), "polygons=", svg.count("<polygon"))

# 提取所有 path 的 d 属性中的坐标，找靠近 U1.6 的填充
# U1.6 板上坐标 (17.137, 5.55)。SVG 通常 1:1 mm 但可能翻转 y。
# 收集所有数字坐标点
nums = re.findall(r'[-+]?\d*\.?\d+', svg)
xs = [float(v) for v in nums]
print("num range x:", min(xs), max(xs))

# 输出前 1500 字符看结构
print("---- head ----")
print(svg[:1500])
