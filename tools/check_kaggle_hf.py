import json
import urllib.request

url = "https://huggingface.co/api/datasets/opencv/opencv_zoo"
try:
    r = urllib.request.urlopen(url, timeout=10)
    data = json.loads(r.read())
    print("Dataset found!")
    print(f"Files: {len(data.get('files', []))}")
    for f in data.get("files", [])[:20]:
        print(f"  {f['rfilename']} ({f.get('size', 0)} bytes)")
except Exception as e:
    print(f"Error: {e}")
