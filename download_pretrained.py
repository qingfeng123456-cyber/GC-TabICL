"""Retrieve or verify the exact upstream checkpoint without changing models."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

HERE=Path(__file__).resolve().parent

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    card=json.loads((HERE/'pretrained/MANIFEST.json').read_text('utf-8'))
    target=HERE/'pretrained'/card['file']
    if target.exists() and target.stat().st_size==card['bytes'] and digest(target)==card['sha256']:
        print('Verified pinned upstream checkpoint:',target.name);return
    if args.verify_only:raise RuntimeError('The checkpoint is missing, incomplete, or a Git LFS pointer. Run this script without --verify-only.')
    url=f'https://huggingface.co/{card["repository"]}/resolve/{card["revision"]}/{card["file"]}'
    temporary=target.with_name(target.name+'.download')
    print('Retrieving pinned upstream checkpoint from Hugging Face.',flush=True)
    with urllib.request.urlopen(url,timeout=180) as response,temporary.open('wb') as output:
        while chunk:=response.read(2**20):output.write(chunk)
    if temporary.stat().st_size!=card['bytes'] or digest(temporary)!=card['sha256']:
        raise RuntimeError('Downloaded checkpoint failed its pinned size or SHA-256 check')
    temporary.replace(target)
    print('Verified checkpoint ready:',target.name)

if __name__=='__main__':main()
