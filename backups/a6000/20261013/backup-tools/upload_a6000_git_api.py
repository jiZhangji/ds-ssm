"""Upload verified Git blobs in batches through gh's authenticated Git Data API.

No credentials are read or written by this script. Existing remote commits must
be ancestors; references are never force-updated. Every blob/tree/commit is
checked against its Git SHA1 before publishing.
"""
import argparse
import base64
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

REPOSITORY = 'jiZhangji/ds-ssm'
BRANCH = 'codex/a6000-backup-20261013-v1'
PREFIX = 'backups/a6000/20261013'

def main(task):
    repo = task / 'repository'
    endpoint = 'repos/' + REPOSITORY
    def git(*args, data=None):
        return subprocess.check_output(['git', '-C', str(repo), *args], input=data)
    def api(method, path, value=None, absent_ok=False):
        args = ['gh', 'api', '--method', method, endpoint + path]
        if value is not None:
            request = task / 'api-request.json'
            request.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
            args += ['--input', str(request)]
        for attempt in range(5):
            p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if p.returncode == 0:
                return json.loads(p.stdout.decode('utf-8'))
            error = p.stderr.decode('utf-8', errors='replace')
            if absent_ok and ('HTTP 404' in error or 'HTTP 409' in error):
                return None
            if attempt == 4:
                raise RuntimeError('GitHub API failure: ' + path + ': ' + error[:500])
            time.sleep(min(2 ** (attempt + 1), 20))
    if api('GET', '')['private']:
        raise ValueError('Expected user-selected public repository')
    head = git('rev-parse', 'HEAD').decode().strip()
    remote = api('GET', '/git/ref/heads/' + BRANCH, absent_ok=True)
    previous_file = task / 'PUBLIC_GIT_RECEIPT.json'
    previous = json.loads(previous_file.read_text()) if previous_file.exists() else None
    if remote and (not previous or remote['object']['sha'] != previous['commit']):
        raise ValueError('Backup branch is not owned by this run; no force update allowed')
    baseline = remote or api('GET', '/git/ref/heads/main')
    parent = baseline['object']['sha']
    parent_commit = api('GET', '/git/commits/' + parent)
    fields = git('show', '-s', '--format=%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI', head).decode().strip().split('\x00')
    author = dict(zip(('name', 'email', 'date'), fields[:3]))
    committer = dict(zip(('name', 'email', 'date'), fields[3:]))
    state_file = task / 'PUBLIC_API_UPLOAD_STATE.json'
    state = json.loads(state_file.read_text()) if state_file.exists() else {'uploaded': {}}
    listing = git('ls-tree', '-r', '-z', head)
    tree, blob_ids = [], []
    for entry in listing.split(b'\0'):
        if not entry:
            continue
        descriptor, path = entry.split(b'\t', 1)
        mode, kind, oid = descriptor.decode().split()
        if kind != 'blob':
            raise ValueError('Unexpected non-blob in backup tree')
        tree.append({'path': path.decode('utf-8'), 'mode': mode, 'type': kind, 'sha': oid})
        blob_ids.append(oid)
    unique = sorted(set(blob_ids) - set(state['uploaded']))
    raw = git('cat-file', '--batch', data=('\n'.join(unique) + '\n').encode()) if unique else b''
    cursor, batch, batch_bytes, completed = 0, [], 0, len(state['uploaded'])
    def flush():
        nonlocal batch, batch_bytes, completed
        if not batch:
            return
        created = api('POST', '/git/trees', {'tree': batch})
        returned = {e['path']: e['sha'] for e in created['tree']}
        for e in batch:
            if returned.get(e['path']) != e['path']:
                raise ValueError('GitHub inline blob SHA mismatch')
            state['uploaded'][e['path']] = returned[e['path']]
        completed += len(batch)
        state_file.write_text(json.dumps(state), encoding='utf-8')
        print('Verified uploaded blobs: ' + str(completed), flush=True)
        batch, batch_bytes = [], 0
    for expected in unique:
        end = raw.index(b'\n', cursor)
        oid, kind, size = raw[cursor:end].decode().split()
        cursor = end + 1
        size = int(size)
        content = raw[cursor:cursor + size]
        cursor += size + 1
        if oid != expected or kind != 'blob':
            raise ValueError('Git batch read mismatch')
        try:
            txt = content.decode('utf-8') if size <= 1500000 else None
        except UnicodeDecodeError:
            txt = None
        if txt is None:
            flush()
            uploaded = api('POST', '/git/blobs', {'content': base64.b64encode(content).decode(), 'encoding': 'base64'})
            if uploaded['sha'] != oid:
                raise ValueError('Binary blob SHA mismatch')
            state['uploaded'][oid] = oid
            completed += 1
            state_file.write_text(json.dumps(state), encoding='utf-8')
            continue
        if len(batch) >= 200 or batch_bytes + size > 2000000:
            flush()
        batch.append({'path': oid, 'mode': '100644', 'type': 'blob', 'content': txt})
        batch_bytes += size
    flush()
    final_tree = api('POST', '/git/trees', {'tree': tree})
    if final_tree['sha'] != git('rev-parse', head + '^{tree}').decode().strip():
        raise ValueError('Final Git tree mismatch')
    public_tree = api('POST', '/git/trees', {'base_tree': parent_commit['tree']['sha'], 'tree': [
        {'path': PREFIX, 'mode': '040000', 'type': 'tree', 'sha': final_tree['sha']} ]})
    baseline_entries = api('GET', '/git/trees/' + parent_commit['tree']['sha'])['tree']
    published_entries = {e['path']: e for e in public_tree['tree'] if '/' not in e['path']}
    for e in baseline_entries:
        if e['path'] != PREFIX.split('/')[0] and published_entries[e['path']]['sha'] != e['sha']:
            raise ValueError('Existing project root entry changed')
    raw_commit = git('cat-file', 'commit', head)
    commit_headers, message = raw_commit.split(b'\n\n', 1)
    other_headers = [line for line in commit_headers.split(b'\n') if not line.startswith((b'tree ', b'parent '))]
    raw_commit = ('tree ' + public_tree['sha'] + '\nparent ' + parent + '\n').encode() + b'\n'.join(other_headers) + b'\n\n' + message
    expected_commit = hashlib.sha1(b'commit ' + str(len(raw_commit)).encode() + b'\0' + raw_commit).hexdigest()
    commit = api('POST', '/git/commits', {'tree': public_tree['sha'], 'parents': [parent],
                 'message': raw_commit.split(b'\n\n', 1)[1].decode(), 'author': author, 'committer': committer})
    if commit['sha'] != expected_commit:
        raise ValueError('GitHub commit SHA mismatch')
    if remote:
        api('PATCH', '/git/refs/heads/' + BRANCH, {'sha': commit['sha'], 'force': False})
    else:
        api('POST', '/git/refs', {'ref': 'refs/heads/' + BRANCH, 'sha': commit['sha']})
    result = {'status': 'PASS', 'repository': REPOSITORY, 'branch': BRANCH, 'prefix': PREFIX,
              'commit': expected_commit, 'tree': public_tree['sha'], 'snapshot_tree': final_tree['sha'],
              'snapshot_commit': head, 'parent': parent, 'verified_blobs': completed, 'paths': len(tree),
              'existing_root_entries_preserved': True}
    previous_file.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, type=Path)
    args = ap.parse_args()
    main(args.task.resolve())
