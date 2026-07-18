import re

path = r'C:\Users\26509\Desktop\workspace\PCBDesign\output\dw01_sch\schematic.kicad_sch'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_prefix = r'C:\Users\26509\Desktop\workspace\PCBDesign\engine\circuits\symbols\battery_protection'
content = content.replace(old_prefix, 'battery_protection')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed lib_id paths')
