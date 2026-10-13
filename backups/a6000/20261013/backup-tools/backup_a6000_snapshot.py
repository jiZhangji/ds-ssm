"""Read-only experiment snapshot collector. Run on a registered server with Python 3.

Copies allowlisted code, protocol manifests, logs, metrics and research figures.
Dataset payloads, model binaries, credentials, environments and unrelated projects
are never read. Original files and running jobs are not modified.
"""
import argparse
import collections
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path('/home/powerleader/ig-ds-ssm')
SKIP_DIRS = {'.git', '.ssh', '.secrets', 'secrets', 'envs', 'env', 'venv',
             '.venv', '__pycache__', 'node_modules', 'site-packages',
             '.cache', 'cache', '.pytest_cache', 'wheelhouse', 'build', 'dist',
             'datasets', 'images', 'masks', 'labels', 'raw', 'raw_data',
             'train', 'val', 'test', 'trainA', 'trainB', 'testA', 'testB',
             'wandb', '.aider.tags.cache.v4'}
TEXT_EXTS = {'.py', '.sh', '.ps1', '.yaml', '.yml', '.json', '.jsonl', '.csv',
             '.tsv', '.md', '.txt', '.log', '.tex', '.bib', '.toml', '.ini',
             '.cfg', '.cpp', '.cu', '.cuh', '.h', '.hpp', '.c', '.cmake',
             '.ipynb', '.html', '.js', '.css', '.xml'}
VISUAL_EXTS = {'.svg', '.pdf', '.png'}
MODEL_EXTS = {'.pt', '.pth', '.ckpt', '.onnx', '.engine', '.safetensors'}
SAFE_NAMES = {'LICENSE', 'Makefile', 'Dockerfile', 'CMakeLists.txt', 'VERSION'}
SENSITIVE_NAME = re.compile(r'(^\.env($|\.)|credential|password|token|private[_-]?key|id_rsa|id_ed25519|cookie|\.pem$|\.key$)', re.I)
SECRET_PATTERNS = [
    ('provider_token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|ms-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})\b')),
    ('private_key', re.compile(r'-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----')),
    ('credential_url', re.compile(r'https?://[^\s/:@]+:[^\s/@]+@')),
    ('literal_assignment', re.compile(r'''(?im)(?:password|passwd|api[_-]?key|access[_-]?token|auth[_-]?token|secret[_-]?key)\s*["']?\s*[:=]\s*["']([^"'\r\n]{4,})["']''')),
    ('password_command', re.compile(r'''(?i)(?:sshpass\s+-p|ConvertTo-SecureString)\s+['"]?[^\s'";]+''')),
    ('bearer_literal', re.compile(r'(?i)Bearer\s+[A-Za-z0-9._~-]{20,}')),
]

def secret_hits(text):
    hits = []
    for category, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            if category == 'literal_assignment':
                value = match.group(1)
                if any(x in value.lower() for x in ('redact', 'placeholder', 'example', 'your_', 'environ', 'getenv', 'token', 'secret', 'password', '{', '$', '<')):
                    continue
            hits.append(category)
    return sorted(set(hits))

def command(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15)
    return p.stdout.decode('utf-8', errors='replace').strip()

def within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False

def collect(out, host, supplement=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    stamp = lambda: datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    meta = {'schema_version': 1, 'host_label': host, 'hostname': command(['hostname']),
            'started_at': stamp(), 'source_root': str(ROOT), 'mode': 'read_only_snapshot',
            'limitations': ['Not an atomic cross-server snapshot.',
                            'Live files may be prefixes of running logs.',
                            'Model binaries and dataset payloads are inventoried, not uploaded.',
                            'Snapshot existence does not verify experiment completion.'],
            'gpu': command(['nvidia-smi', '--query-gpu=index,name,memory.used,utilization.gpu', '--format=csv,noheader']),
            'disk': command(['df', '-h', str(ROOT)]),
            'roots': [], 'excluded_dirs': [], 'excluded_counts': {}, 'included_count': 0}
    roots = [p for p in ROOT.iterdir() if p.name not in SKIP_DIRS and not p.name.startswith('.')]
    extra = Path('/home/powerleader/codex_tasks/complexity_20261011')
    if extra.exists():
        roots.append(extra)
    if supplement:
        pending = json.loads(Path(supplement).read_text(encoding='utf-8'))
        roots = [ROOT / row['path'] for row in pending if row['reason'] == 'larger_than_20MiB']
        meta['supplement_of'] = str(supplement)
    records, models, warnings = [], [], []
    counts = collections.Counter()
    with tarfile.open(out / 'snapshot.tar.gz', 'w:gz', compresslevel=1) as archive:
        def examine(path):
            if path.is_symlink():
                counts['symlink_not_followed'] += 1
                return
            try:
                s = path.stat()
            except OSError:
                counts['stat_failed'] += 1
                return
            rel = path.relative_to(ROOT) if within(path, ROOT) else Path('external/complexity_20261011') / path.relative_to(extra)
            rel = rel.as_posix()
            if SENSITIVE_NAME.search(path.name):
                counts['credential_filename'] += 1
                return
            suffix = path.suffix.lower()
            if suffix in MODEL_EXTS:
                models.append({'path': rel, 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'sha256': None, 'status': 'not_read_not_uploaded'})
                return
            visual = suffix in VISUAL_EXTS
            if suffix not in TEXT_EXTS | VISUAL_EXTS and path.name not in SAFE_NAMES:
                counts['extension'] += 1
                return
            if visual and any(part in {'data', 'cross_data', 'taskdata', 'asserts', 'samples', 'patches', 'tiles', 'JPEGImages', 'Annotations'} for part in path.parts):
                counts['dataset_visual'] += 1
                return
            if s.st_size > (128 if supplement else 20) * 1024 * 1024:
                counts['larger_than_20MiB'] += 1
                warnings.append({'path': rel, 'reason': 'larger_than_20MiB', 'bytes': s.st_size})
                return
            try:
                with path.open('rb') as f:
                    data = f.read(s.st_size)
                after = path.stat()
            except OSError:
                counts['read_failed'] += 1
                warnings.append({'path': rel, 'reason': 'read_failed'})
                return
            if suffix in TEXT_EXTS or path.name in SAFE_NAMES:
                try:
                    txt = data.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        txt = data.decode('gb18030')
                    except UnicodeDecodeError:
                        counts['undecodable_text'] += 1
                        warnings.append({'path': rel, 'reason': 'undecodable_text'})
                        return
                hits = secret_hits(txt)
                if hits:
                    counts['possible_secret_content'] += 1
                    warnings.append({'path': rel, 'reason': 'possible_secret_content', 'categories': hits})
                    return
            changed = (s.st_size, s.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            info = tarfile.TarInfo('payload/' + rel)
            info.size, info.mode, info.mtime = len(data), s.st_mode & 0o777, s.st_mtime
            archive.addfile(info, io.BytesIO(data))
            records.append({'path': rel, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                            'mode': s.st_mode & 0o777, 'source_mtime_ns': s.st_mtime_ns,
                            'changed_during_read': changed})
        for root in sorted(roots):
            meta['roots'].append(str(root))
            if root.is_symlink():
                counts['symlink_root'] += 1
                continue
            if root.is_file():
                examine(root)
                continue
            for base, dirs, files in os.walk(root, followlinks=False):
                dropped = [d for d in dirs if d in SKIP_DIRS or d.endswith('.egg-info') or SENSITIVE_NAME.search(d) or (Path(base) / d).is_symlink()]
                for d in dropped:
                    meta['excluded_dirs'].append(str((Path(base) / d).relative_to(ROOT)) if within(Path(base) / d, ROOT) else str(Path(base) / d))
                dirs[:] = sorted(d for d in dirs if d not in dropped)
                for f in sorted(files):
                    examine(Path(base) / f)
    meta.update(finished_at=stamp(), included_count=len(records), included_bytes=sum(r['bytes'] for r in records),
                model_inventory_count=len(models), model_inventory_bytes=sum(r['bytes'] for r in models),
                live_changed_count=sum(r['changed_during_read'] for r in records), excluded_counts=dict(counts))
    for name, value in [('metadata.json', meta), ('files.json', records), ('models_not_uploaded.json', models), ('warnings.json', warnings)]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    digest = hashlib.sha256((out / 'snapshot.tar.gz').read_bytes()).hexdigest()
    (out / 'SHA256SUMS').write_text(digest + '  snapshot.tar.gz\n', encoding='ascii')
    print(json.dumps({'host': host, 'files': len(records), 'bytes': meta['included_bytes'],
                      'archive_bytes': (out / 'snapshot.tar.gz').stat().st_size, 'sha256': digest,
                      'warnings': len(warnings), 'models_not_uploaded': len(models)}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--supplement', help='Warnings from a previous snapshot; copy its oversized records.')
    args = parser.parse_args()
    os.nice(15)
    collect(args.out, args.host, args.supplement)
