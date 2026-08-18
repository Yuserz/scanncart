#!/usr/bin/env python3
"""Upload remaining images to Roboflow."""
import os, sys, time, argparse
from dotenv import load_dotenv

load_dotenv()

def upload_class(project, class_name, folder):
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    batch_name = f'scanncart-{class_name}'
    print(f'  {len(files)} images -> {batch_name}')
    
    uploaded = 0
    for i, fname in enumerate(files):
        img_path = os.path.join(folder, fname)
        try:
            project.upload(image_path=img_path, batch_name=batch_name, split='train')
            uploaded += 1
        except Exception as e:
            if 'duplicate' not in str(e).lower():
                print(f'    Error: {fname}: {e}')
        if (i + 1) % 25 == 0:
            print(f'    {i+1}/{len(files)}')
            time.sleep(0.3)
    
    print(f'  Done: {uploaded} processed')
    return uploaded

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cls', '-c', help='Upload specific class only')
    args = parser.parse_args()
    
    api_key = os.getenv('ROBOFLOW_API_KEY')
    project_id = os.getenv('ROBOFLOW_PROJECT_ID')
    
    if not api_key or not project_id:
        print("Error: Check .env file for ROBOFLOW_API_KEY and ROBOFLOW_PROJECT_ID")
        return 1
    
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace('yusri-caloyloy').project(project_id)
    
    classes = {
        '555-sardines': 'cleaned/555-sardines',
        'bear-brand-milk': 'cleaned/bear-brand-milk',
        'century-tuna': 'cleaned/century-tuna',
        'lucky-me-pancit': 'cleaned/lucky-me-pancit',
        'safeguard': 'cleaned/safeguard',
        'silver-swan-vinegar': 'cleaned/silver-swan-vinegar'
    }
    
    if args.cls:
        if args.cls not in classes:
            print(f"Unknown class. Use: {', '.join(classes.keys())}")
            return 1
        classes = {args.cls: classes[args.cls]}
    
    print("Uploading to Roboflow...\n")
    total = 0
    for cls, folder in classes.items():
        if os.path.exists(folder):
            total += upload_class(project, cls, folder)
    
    print(f"\nTotal: {total} images processed")
    return 0

if __name__ == '__main__':
    sys.exit(main())
