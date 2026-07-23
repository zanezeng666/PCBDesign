import json, urllib.request, urllib.parse

data = urllib.parse.urlencode({'frame_w_mm': 60, 'frame_h_mm': 30}).encode()
req = urllib.request.Request('http://localhost:8000/api/simulate', data=data, method='POST')
with urllib.request.urlopen(req, timeout=30) as resp:
    result = json.loads(resp.read())

# Print the full steps for each side
for step in result.get('steps', []):
    side = step['image'].replace('.jpg', '')
    print(f"\n=== {step['image']} ===")
    for key in sorted(step.keys()):
        val = step[key]
        if key in ('detection_image_base64', 'annotated_image_base64', 'rectified_png_base64'):
            print(f"  {key}: <base64: {len(str(val))} chars>")
        else:
            print(f"  {key}: {val}")

print(f"\n=== Summary ===")
print(f"  dual_ok: {result.get('dual_ok')}")
print(f"  frame_w_mm: {result.get('frame_w_mm')}")
print(f"  frame_h_mm: {result.get('frame_h_mm')}")
