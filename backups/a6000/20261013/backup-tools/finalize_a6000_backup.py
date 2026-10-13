"""Create release assets, scan the separate repository, or verify GitHub receipts."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('collector', HERE / 'backup_a6000_snapshot.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
REPOSITORY = 'jiZhangji/ds-ssm'
TAG = 'a6000-snapshot-20261013-v1'
BRANCH = 'codex/a6000-backup-20261013-v1'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def gh(*args):
    return json.loads(subprocess.check_output(['gh', *args]).decode('utf-8'))

def main(task, mode):
    repo = task / 'repository'
    index = json.loads((repo / 'SNAPSHOT_INDEX.json').read_text(encoding='utf-8'))
    if mode == 'assets':
        target = task / 'release-assets'
        target.mkdir(exist_ok=True)
        for row in index:
            files = [(task / 'downloads' / row['host'].lower() / row.get('local_archive_name', 'snapshot.tar.gz'), row['release_asset'])]
            if 'large_records' in row:
                files.append((task / 'downloads' / row['host'].lower() / 'large_records' / 'snapshot.tar.gz', row['large_records']['release_asset']))
            for src, name in files:
                if not (target / name).exists():
                    os.link(src, target / name)
        (target / 'SNAPSHOT_INDEX.json').write_bytes((repo / 'SNAPSHOT_INDEX.json').read_bytes())
        (target / 'VERIFICATION.json').write_bytes((repo / 'VERIFICATION.json').read_bytes())
        sums = '\n'.join(digest(p) + '  ' + p.name for p in sorted(target.iterdir()) if p.is_file() and p.name != 'SHA256SUMS') + '\n'
        (target / 'SHA256SUMS').write_text(sums, encoding='ascii')
        (task / 'release-body.md').write_text('''按用户要求保存 A/B/C/D/F 的 Mamba / IG-DS-SSM 实验代码、配置、逐轮账本、训练与评估记录、正负结果、图表和 PDF。

五台服务器分时只读快照；每个压缩包及包内所有文件均已核 SHA256。长日志与大型审计记录另附 C/F large-records 补充包。可浏览源码与结果位于仓库 servers/，项目说明与论文源码位于 project-notes/；RESULTS_INDEX.csv 提供结果路径索引。

仍在训练的运行只保存快照时已有记录，不代表整批完成。凭据、环境、数据集图像及模型二进制未上传；权重仅有路径/大小清单。未修改现有队列或正式实验文件；本轮未建立定时任务。
''', encoding='utf-8')
        print(json.dumps({'assets': len(list(target.iterdir())), 'bytes': sum(p.stat().st_size for p in target.iterdir())}))
    elif mode == 'scan':
        if os.name == 'nt':
            repo = Path('\\\\?\\' + str(repo))
        issues, scanned, largest, examined = [], 0, 0, 0
        seen = set()
        for base, dirs, files in os.walk(repo):
            dirs[:] = [d for d in dirs if d != '.git']
            for name in files:
                path = Path(base) / name
                data = path.read_bytes()
                largest = max(largest, len(data))
                if path.suffix.lower() in collector.TEXT_EXTS or name in collector.SAFE_NAMES:
                    examined += 1
                    content_hash = hashlib.sha256(data).digest()
                    if content_hash in seen:
                        continue
                    seen.add(content_hash)
                    hits = collector.secret_hits(data.decode('utf-8', errors='replace'))
                    scanned += 1
                    if hits:
                        issues.append({'path': path.relative_to(repo).as_posix(), 'categories': hits})
        report = {'status': 'PASS' if not issues else 'FAILED', 'text_files_examined': examined, 'unique_text_contents_scanned': scanned, 'largest_file_bytes': largest, 'issues': issues}
        (task / 'SECRET_SCAN.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))
        if issues:
            raise SystemExit(1)
    else:
        release = gh('api', 'repos/' + REPOSITORY + '/releases/tags/' + TAG)
        visibility = gh('api', 'repos/' + REPOSITORY)['private']
        head = gh('api', 'repos/' + REPOSITORY + '/git/ref/heads/' + BRANCH)['object']['sha']
        local_head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
        git_receipt = json.loads((task / 'PUBLIC_GIT_RECEIPT.json').read_text())
        if visibility or git_receipt['commit'] != head or git_receipt['snapshot_commit'] != local_head:
            raise ValueError('Visibility or branch receipt mismatch')
        subtree = gh('api', 'repos/' + REPOSITORY + '/git/commits/' + head)['tree']['sha']
        for component in git_receipt['prefix'].split('/'):
            tree = gh('api', 'repos/' + REPOSITORY + '/git/trees/' + subtree)
            subtree = next(e['sha'] for e in tree['tree'] if e['path'] == component and e['type'] == 'tree')
        if subtree != git_receipt['snapshot_tree']:
            raise ValueError('Published snapshot subtree differs')
        remote_assets = {a['name']: a for a in release['assets']}
        receipt = []
        for path in sorted((task / 'release-assets').iterdir()):
            remote = remote_assets[path.name]
            expected = digest(path)
            if remote['size'] != path.stat().st_size:
                raise ValueError('Asset byte count mismatch: ' + path.name)
            server_digest = remote.get('digest')
            if server_digest:
                if server_digest != 'sha256:' + expected:
                    raise ValueError('GitHub digest mismatch: ' + path.name)
                verification = 'GitHub asset SHA256'
            else:
                roundtrip = task / 'roundtrip'
                roundtrip.mkdir(exist_ok=True)
                subprocess.run(['gh', 'release', 'download', TAG, '--repo', REPOSITORY, '--pattern', path.name, '--dir', str(roundtrip)], check=True)
                if digest(roundtrip / path.name) != expected:
                    raise ValueError('Downloaded asset SHA256 mismatch')
                verification = 'downloaded SHA256'
            receipt.append({'name': path.name, 'bytes': remote['size'], 'sha256': expected, 'verification': verification})
        result = {'status': 'PASS', 'repository': REPOSITORY, 'private': visibility, 'commit': head,
                  'branch': BRANCH, 'snapshot_prefix': git_receipt['prefix'],
                  'release_url': release['html_url'], 'tag': TAG, 'assets': receipt,
                  'snapshot_payload_files_verified': sum(r['files_verified'] + r.get('large_records', {}).get('files_verified', 0) for r in index),
                  'browsable_server_files': sum(r['browsable_files'] for r in index)}
        (task / 'UPLOAD_RECEIPT.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['assets', 'scan', 'verify'])
    ap.add_argument('--task', required=True, type=Path)
    args = ap.parse_args()
    main(args.task.resolve(), args.mode)
