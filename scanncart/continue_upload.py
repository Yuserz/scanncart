"""Continue uploading remaining images to Roboflow."""
from dotenv import load_dotenv
import os, requests, time
from pathlib import Path
from roboflow import Roboflow

load_dotenv(Path(__file__).resolve().parent / ".env")
api_key = os.getenv('ROBOFLOW_API_KEY')
project_id = os.getenv('ROBOFLOW_PROJECT_ID')
CLEANED_DIR = Path(__file__).resolve().parent / "cleaned"

rf = Roboflow(api_key=api_key)
workspace = rf.workspace()
project = workspace.project(project_id)
batches = project.get_batches()

class_batches = {}
for b in batches.get('batches', []):
    name = b.get('name', '?')
    count = b.get('images', 0)
    if name.startswith('scanncart-'):
        cls = name.replace('scanncart-', '')
        class_batches[cls] = class_batches.get(cls, 0) + count

print("=== Current Status ===")
for cls in ['555-sardines', 'bear-brand-milk', 'century-tuna', 'lucky-me-pancit', 'safeguard', 'silver-swan-vinegar']:
    u = class_batches.get(cls, 0)
    e = 324 if cls == 'bear-brand-milk' else 300
    s = 'OK' if u >= e else f'MISSING {e-u}'
    print(f"  {cls}: {u}/{e} ({s})")

print("\n=== Uploading Remaining ===")
url = f'https://api.roboflow.com/dataset/{project_id}/upload'
uploaded = 0

for cls in ['century-tuna', 'lucky-me-pancit', 'safeguard', 'silver-swan-vinegar']:
    cls_dir = CLEANED_DIR / cls
    images = sorted(cls_dir.glob("*.jpg"))
    already = class_batches.get(cls, 0)
    remaining = images[already:]
    if not remaining:
        print(f"  {cls}: Complete!")
        continue
    print(f"  {cls}: {len(remaining)} remaining")
    for img in remaining:
        params = {'api_key': api_key, 'name': img.name, 'split': 'train', 'batch': f'scanncart-{cls}'}
        with open(img, 'rb') as f:
            files = {'file': (img.name, f, 'image/jpeg')}
            resp = requests.post(url, params=params, files=files)
        result = resp.json()
        if result.get('success') or result.get('duplicate'):
            uploaded += 1
        time.sleep(0.15)

print(f"\nUploaded: {uploaded} additional images")
print("Done!")
