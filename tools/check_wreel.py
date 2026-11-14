import json
import zipfile

with zipfile.ZipFile(r"D:\new_IDEA\worldreel\runs\generated\S010.wreel") as z:
    manifest = json.loads(z.read("manifest.json"))
    print("format_version:", manifest.get("format_version"))
    print("scene.layers:", json.dumps(manifest.get("scene", {}).get("layers", []), indent=2))
    print("actors count:", len(manifest.get("actors", [])))
    for a in manifest.get("actors", []):
        geom = a.get("geometry", {})
        lods = geom.get("lods", [{}])
        print(
            "  {}: kind={}, path={}".format(
                a.get("id"), geom.get("kind"), lods[0].get("path", "N/A")
            )
        )
    print("audio.tracks count:", len(manifest.get("audio", {}).get("tracks", [])))
