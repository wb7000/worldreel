import json
import zipfile

with zipfile.ZipFile(r"D:\new_IDEA\worldreel\runs\generated\S010.wreel") as z:
    manifest = json.loads(z.read("manifest.json"))
    files = manifest.get("files", {})
    print("files count:", len(files))
    for name, info in sorted(files.items()):
        print(f"  {name}: {info.get('bytes')} bytes, {info.get('media_type')}")
