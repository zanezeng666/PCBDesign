"""精确扫描 Q1.1 焊盘左缘的 zone 覆盖边界。"""
import pcbnew

b = pcbnew.LoadBoard("data/ic_templates/DW01-G/pcb.kicad_pcb")
z = b.Zones()[0]
filler = pcbnew.ZONE_FILLER(b)
filler.Fill([z])

def hit(x, y):
    v = pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))
    return z.HitTestFilledArea(pcbnew.F_Cu, v)

# Q1.1 pad x∈[12.40,13.87], y∈[12.825,13.225]，中心 (13.137,13.025)
print("== 沿 y=13.025 水平扫描 x（找 zone 边界）==")
for x10 in range(118, 140, 1):  # x from 11.8 to 13.9 step 0.1
    x = x10 / 10.0
    print(f"  x={x:.1f}: {hit(x, 13.025)}")

print("== 沿 x=13.137 垂直扫描 y ==")
for y10 in range(124, 138, 1):
    y = y10 / 10.0
    print(f"  y={y:.1f}: {hit(13.137, y)}")

# 检查 Q1.1 焊盘属性
for fp in b.GetFootprints():
    if fp.GetReference() == "Q1":
        p = fp.FindPadByNumber("1")
        print("Q1.1 net=", p.GetNetname(), "attr=", p.GetAttribute(),
              "size=", pcbnew.ToMM(p.GetSize().x), pcbnew.ToMM(p.GetSize().y),
              "layers=", [pcbnew.EDACore.ToLAYER_ID(l) if False else l for l in p.GetLayerSet().Seq()])
