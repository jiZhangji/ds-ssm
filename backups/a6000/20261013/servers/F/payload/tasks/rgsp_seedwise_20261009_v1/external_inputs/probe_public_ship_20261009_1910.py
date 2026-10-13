#!/usr/bin/env python3
"""Read public source metadata or download the exact hash-bound Ship archive.

No credentials, signed URLs, installation, dataset publication, or training.
An alternate transport cannot change the required source bytes.
"""
import argparse
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

TASK = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1')
SOURCE = 'https://huggingface.co/datasets/shimiandeshu/sar-m/resolve/main/ship_dataset_v0.zip'
API = 'https://huggingface.co/api/datasets/shimiandeshu/sar-m/tree/main'
EXPECTED_SHA = '6a185fbf2f51408d19ca941745581a8e1aabbacd33db6494985621086495481e'
EXPECTED_SIZE = 426990175


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', type=Path)
    args = parser.parse_args()
    if socket.gethostname() not in ('a6000-c', 'powerleader-a6000-f'):
        raise RuntimeError('This transport is restricted to the two audited Ship hosts')
    started = time.monotonic()
    try:
        with urllib.request.urlopen(API, timeout=20) as response:
            listing = json.load(response)
        entries = [row for row in listing if row.get('path') == 'ship_dataset_v0.zip']
        if len(entries) != 1:
            raise RuntimeError('Public source does not have the exact named artifact')
        entry = entries[0]
        lfs = entry.get('lfs', {})
        meta = {'size': entry.get('size'), 'lfs_sha256': lfs.get('oid'),
                'expected_size': EXPECTED_SIZE, 'expected_sha256': EXPECTED_SHA}
        if meta['size'] != EXPECTED_SIZE or meta['lfs_sha256'] != EXPECTED_SHA:
            raise RuntimeError('Public source metadata differs from the reviewed source')
        print(json.dumps({'status': 'public_source_metadata_verified', **meta}), flush=True)
        if args.download is None:
            return
        target = args.download.resolve()
        allowed = (TASK / 'external_inputs/raw_archives').resolve()
        if target.parent != allowed or target.exists():
            raise RuntimeError('Download must use a fresh explicit archive filename')
        digest = hashlib.sha256()
        count = 0
        with urllib.request.urlopen(SOURCE, timeout=30) as response:
            with target.open('xb') as outgoing:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    outgoing.write(block)
                    digest.update(block)
                    count += len(block)
                    if count > EXPECTED_SIZE:
                        raise RuntimeError('Public source exceeded the expected archive size')
                    if count % (32 * 1024 * 1024) == 0:
                        print(json.dumps({'received_bytes': count, 'elapsed_seconds': time.monotonic() - started}), flush=True)
        if count != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_SHA:
            raise RuntimeError('Downloaded source failed size/hash verification; preserved, never published')
        print(json.dumps({'status': 'verified_download', 'path': str(target),
                          'bytes': count, 'sha256': digest.hexdigest(),
                          'elapsed_seconds': time.monotonic() - started}), flush=True)
    except urllib.error.HTTPError as error:
        print(json.dumps({'status': 'transport_unavailable', 'http_status': error.code}), flush=True)
        raise SystemExit(2)
    except (urllib.error.URLError, TimeoutError) as error:
        # Do not log redirect URLs, which may contain expiring signed tokens.
        print(json.dumps({'status': 'transport_unavailable', 'error_type': type(error).__name__}), flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
