#!/usr/bin/env python3
"""Download only the exact public Ship ZIP; never stage or publish readiness.

The public mirror API returned 403, while the public file HEAD returned 200
with the exact expected size. Integrity therefore comes from the original
locally verified full-file SHA256, not from mirror metadata or a redirect URL.
No credentials or redirect URLs are read, printed, or stored.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import urllib.error
import urllib.request

ROOT = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_seedwise_20261009_v1')
SIZE = 426990175
SHA256 = '6a185fbf2f51408d19ca941745581a8e1aabbacd33db6494985621086495481e'
SOURCE = 'https://hf-mirror.com/datasets/shimiandeshu/sar-m/resolve/main/ship_dataset_v0.zip'


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True, type=Path)
    parser.add_argument('--expected-hostname', required=True, choices=['a6000-c', 'powerleader-a6000-f'])
    args = parser.parse_args()
    if os.name != 'posix' or socket.gethostname() != args.expected_hostname:
        raise RuntimeError('Only the explicitly assigned existing server may download')
    target = args.target
    if target != target.resolve() or target.parent != ROOT / 'external_inputs/raw_archives':
        raise RuntimeError('Only an explicit fresh file in the task archive directory is allowed')
    if not target.name.startswith('ship_dataset_v0.zip.mirror_') or target.exists():
        raise RuntimeError('Target must be a fresh mirror attempt; partials are never overwritten')
    record = target.with_name(target.name + '.verified.json')
    if record.exists():
        raise RuntimeError('Completion record already exists')
    started = time.monotonic()
    request = urllib.request.Request(SOURCE, headers={'User-Agent': 'Mozilla/5.0'})
    hasher = hashlib.sha256()
    count = 0
    emit({'status': 'starting', 'target': str(target), 'expected_bytes': SIZE,
          'expected_sha256': SHA256, 'hostname': socket.gethostname()})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError('Public download did not return a complete response')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != SIZE:
                raise RuntimeError('Public response size differs from pinned archive')
            with target.open('xb') as outgoing:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    count += len(block)
                    if count > SIZE:
                        raise RuntimeError('Download exceeded pinned archive length')
                    outgoing.write(block)
                    hasher.update(block)
                    if count % (32 * 1024 * 1024) == 0:
                        emit({'status': 'downloading', 'bytes': count,
                              'elapsed_seconds': time.monotonic() - started})
                outgoing.flush()
                os.fsync(outgoing.fileno())
        if count != SIZE or hasher.hexdigest() != SHA256:
            raise RuntimeError('Downloaded file failed pinned size/SHA; preserved, not usable')
        result = {'status': 'verified_download', 'hostname': socket.gethostname(),
                  'path': str(target), 'bytes': count, 'sha256': hasher.hexdigest(),
                  'source': SOURCE, 'elapsed_seconds': time.monotonic() - started,
                  'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'readiness_published': False}
        with record.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit(result)
    except Exception as error:
        failed = {'status': 'failed_preserved', 'bytes': count,
                  'error_type': type(error).__name__, 'readiness_published': False}
        if isinstance(error, urllib.error.HTTPError):
            failed['http_status'] = error.code
        emit(failed)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
