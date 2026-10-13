"""Verify downloaded snapshots and prepare a separate, browsable Git repository."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('collector', HERE / 'backup_a6000_snapshot.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def prepare(task, project):
    repo = task / 'repository'
    if os.name == 'nt':
        repo = Path('\\\\?\\' + str(repo))
        project = Path('\\\\?\\' + str(project))
    summary = []
    result_refs = []
    for host in 'abcdf':
        src = task / 'downloads' / host
        metadata = json.loads((src / 'metadata.json').read_text(encoding='utf-8'))
        entries = json.loads((src / 'files.json').read_text(encoding='utf-8'))
        expected = (src / 'SHA256SUMS').read_text().split()[0]
        archive = src / 'snapshot.tar.gz'
        alternate = src / 'snapshot.lan.tar.gz'
        if alternate.exists() and sha(alternate.read_bytes()) == expected:
            archive = alternate
        if sha(archive.read_bytes()) != expected:
            raise ValueError('Archive SHA256 mismatch: ' + host)
        mapping = {r['path']: r for r in entries}
        target = repo / 'servers' / host.upper()
        target.mkdir(parents=True, exist_ok=True)
        for name in ('metadata.json', 'files.json', 'models_not_uploaded.json', 'warnings.json', 'SHA256SUMS'):
            shutil.copyfile(src / name, target / name)
        count, published, bytes_published = 0, 0, 0
        with tarfile.open(archive) as tf:
            for member in tf:
                if not member.isfile() or not member.name.startswith('payload/'):
                    raise ValueError('Unsafe archive member')
                rel = member.name[len('payload/'):]
                if '..' in Path(rel).parts or Path(rel).is_absolute():
                    raise ValueError('Unsafe archive path')
                data = tf.extractfile(member).read()
                entry = mapping[rel]
                if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
                    raise ValueError('Payload SHA256 mismatch: ' + host + '/' + rel)
                count += 1
                if Path(rel).name.lower() in {'result.json', 'eval_test.json', 'train_complete.json', 'test_metrics.json', 'completion.json', 'epochs.jsonl', 'results.csv', 'history.csv', 'summary.json', 'evaluation_complete.json', 'epoch_ledger.jsonl'}:
                    result_refs.append({'host': host.upper(), 'source_path': rel, 'sha256': entry['sha256'], 'bytes': len(data), 'release_asset': 'A6000-' + host.upper() + '-snapshot-20261013.tar.gz'})
                suffix = Path(rel).suffix.lower()
                # Full figures, dataset audit exports and long logs live in the release.
                if suffix in collector.VISUAL_EXTS or len(data) > 2 * 1024 * 1024 or any(p in {'extracted', 'data', 'cross_data', 'taskdata', '_raw_airship_xml'} for p in Path(rel).parts):
                    continue
                destination = target / 'payload' / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                published += 1
                bytes_published += len(data)
        if count != len(entries):
            raise ValueError('Missing snapshot members: ' + host)
        row = {'host': host.upper(), 'files_verified': count, 'browsable_files': published,
               'browsable_bytes': bytes_published, 'snapshot_bytes': metadata['included_bytes'],
               'archive_bytes': archive.stat().st_size, 'archive_sha256': expected,
               'local_archive_name': archive.name,
               'started_at': metadata['started_at'], 'finished_at': metadata['finished_at'],
               'live_changed_count': metadata['live_changed_count'],
               'weight_files_not_uploaded': metadata['model_inventory_count'],
               'weight_bytes_not_uploaded': metadata['model_inventory_bytes'],
               'release_asset': 'A6000-' + host.upper() + '-snapshot-20261013.tar.gz'}
        supplement = src / 'large_records'
        if supplement.exists():
            small_entries = json.loads((supplement / 'files.json').read_text(encoding='utf-8'))
            small_sha = (supplement / 'SHA256SUMS').read_text().split()[0]
            small_archive = supplement / 'snapshot.tar.gz'
            if sha(small_archive.read_bytes()) != small_sha:
                raise ValueError('Supplement archive mismatch')
            with tarfile.open(small_archive) as tf:
                supplemental_map = {r['path']: r for r in small_entries}
                for member in tf:
                    data = tf.extractfile(member).read()
                    if sha(data) != supplemental_map[member.name[len('payload/'):]]['sha256']:
                        raise ValueError('Supplement payload mismatch')
            supplemental_target = target / 'large_records'
            supplemental_target.mkdir(exist_ok=True)
            for name in ('metadata.json', 'files.json', 'warnings.json', 'SHA256SUMS'):
                shutil.copyfile(supplement / name, supplemental_target / name)
            row['large_records'] = {'files_verified': len(small_entries), 'archive_sha256': small_sha,
                                    'archive_bytes': small_archive.stat().st_size,
                                    'release_asset': 'A6000-' + host.upper() + '-large-records-20261013.tar.gz'}
        summary.append(row)
        print(json.dumps(row), flush=True)
    # Preserve project-level explanations, not workstation secrets or caches.
    docs = []
    for name in ('README_CURRENT.md', 'HANDOFF_FOR_NEXT_MODEL.md'):
        docs.append(project / name)
    for name in ('reports', 'database', 'docs/project-memory', 'paper_draft'):
        base = project / name
        docs.extend(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in {'.md', '.csv', '.tex', '.bib', '.json'} and 'archive' not in p.parts and 'snapshots' not in p.parts)
    local_records, excluded = [], []
    for path in docs:
        rel = path.relative_to(project)
        if path.stat().st_size > 2 * 1024 * 1024 or collector.SENSITIVE_NAME.search(path.name):
            excluded.append({'path': rel.as_posix(), 'reason': 'size_or_sensitive_name'})
            continue
        data = path.read_bytes()
        hits = collector.secret_hits(data.decode('utf-8', errors='replace'))
        if hits:
            excluded.append({'path': rel.as_posix(), 'reason': 'possible_secret', 'categories': hits})
            continue
        dst = repo / 'project-notes' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        local_records.append({'path': rel.as_posix(), 'bytes': len(data), 'sha256': sha(data)})
    (repo / 'project-notes' / 'FILES.json').write_text(json.dumps(local_records, ensure_ascii=False, indent=2), encoding='utf-8')
    (repo / 'project-notes' / 'EXCLUSIONS.json').write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding='utf-8')
    (repo / 'SNAPSHOT_INDEX.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    with (repo / 'RESULTS_INDEX.csv').open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=['host', 'source_path', 'sha256', 'bytes', 'release_asset'])
        writer.writeheader()
        writer.writerows(result_refs)
    (task / 'VERIFICATION.json').write_text(json.dumps({'status': 'PASS', 'method': 'archive and every payload SHA256 verified', 'hosts': summary, 'local_notes_files': len(local_records), 'local_notes_excluded': excluded}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (repo / 'VERIFICATION.json').write_bytes((task / 'VERIFICATION.json').read_bytes())
    (repo / 'backup-tools').mkdir(exist_ok=True)
    for name in ('backup_a6000_snapshot.py', 'prepare_a6000_backup.py', 'finalize_a6000_backup.py', 'upload_a6000_git_api.py'):
        shutil.copyfile(HERE / name, repo / 'backup-tools' / name)
    rows = '\n'.join('| {host} | {files_verified} | {browsable_files} | {archive_mb:.1f} | {started_at} |'.format(archive_mb=r['archive_bytes'] / 1e6, **r) for r in summary)
    readme = '''# A6000 Mamba / IG-DS-SSM 实验备份

按用户明确选择，上传至现有公开仓库的独立备份分支；2026-10-13 保存 A/B/C/D/F 的当前实验代码、记录和结果。

## 查找文件

- `servers/<主机>/payload/work_train_single/`：各机实际源代码版本。
- `servers/<主机>/payload/tasks/`：冻结源码、实验计划、协议、逐轮账本、完成回执和指标。
- `servers/<主机>/payload/experiments/`：历史跨任务结果及负结果。
- `project-notes/`：本地项目报告、结果表、论文源码和正式记忆的本次只读副本。
- `RESULTS_INDEX.csv`：训练/评估结果、完成记录和逐轮账本的路径索引；不把存在某文件等同实验验收。
- [本次 Release](https://github.com/jiZhangji/ds-ssm/releases/tag/a6000-snapshot-20261013-v1)：完整代码/记录快照、图表/PDF、长日志及大型数据审计记录。

| 主机 | 快照文件（逐项核 SHA） | 可浏览文件 | 压缩包 MB | 开始时刻 +08:00 |
|---|---:|---:|---:|---|
''' + rows + '''

## 证据与恢复

`SNAPSHOT_INDEX.json` 给出 Release 资产与 SHA256；各主机 `files.json` 列出每个原路径、字节数、SHA256、读到时的 mtime 和读取期间是否变化。下载对应快照后校验包 SHA256，解压 `payload/` 即可恢复文件；再按 `files.json` 核逐项 SHA256。不同主机的同名源码保持独立，避免覆盖实际运行版本。

这是分时只读快照；仍在运行的实验包含当时已有记录，不代表训练完成或新的成绩验收。未改训练队列、未运行模型、未停止既有进程。

## 排除范围

不上传凭据、私钥、认证目录、环境和缓存、数据集图像及模型二进制。`models_not_uploaded.json` 仅登记权重路径、大小和 mtime，**这些权重没有上传，未计算其 SHA256**；不等同完整灾备。扫描排除与大小限制见 `warnings.json`、`metadata.json` 和 `project-notes/EXCLUSIONS.json`；C/F 被大小限制跳过的重要记录通过 `large_records/` 与 Release 补充包保存。原始服务器文件保留。

## 更新

本轮为按需执行的一次备份；没有建立定时任务。`backup-tools/` 保存本次采集/核验脚本，不含认证信息。重新执行应使用新的唯一任务目录及快照版本。
'''
    (repo / 'README.md').write_text(readme, encoding='utf-8')
    (repo / '.gitignore').write_text('__pycache__/\n*.pyc\n*.tar.gz\n', encoding='utf-8')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, type=Path)
    ap.add_argument('--project', required=True, type=Path)
    args = ap.parse_args()
    prepare(args.task.resolve(), args.project.resolve())
