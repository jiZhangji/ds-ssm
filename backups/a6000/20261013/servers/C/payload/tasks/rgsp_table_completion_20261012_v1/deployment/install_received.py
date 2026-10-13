"""Install hash-bound task archives into new server-only directories."""
import hashlib
import json
import os
from pathlib import Path
import socket
import tarfile
from datetime import datetime, timezone

root = Path('/home/powerleader/ig-ds-ssm/tasks/rgsp_table_completion_20261012_v1')
if os.name != 'posix' or Path(__file__).resolve().parent != root / 'deployment':
    raise RuntimeError('Explicit task deployment directory on Linux required')
packages = [
    ('runner_deploy_v2.tar.gz', 'd098c5425db2120cc4fa6f5731d52c14363d5502a83138863577f9da7220e9fe', root / 'training'),
    ('ssdd_official_928_232_v1.tar.gz', 'ecf9ad32302e97ba6fbc477722b5d8f9f7fc95f58f421cbc70d6e9d5b596cc02', root / 'data/SSDD'),
]
verified = []
for name, digest, target in packages:
    source = root / 'deployment' / name
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != digest or target.exists():
        raise RuntimeError('Archive hash mismatch or destination already exists: ' + name)
    with tarfile.open(source, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            relative = Path(member.name)
            resolved = (target / relative).resolve()
            if (relative.is_absolute() or '..' in relative.parts or
                    target.resolve() not in resolved.parents and resolved != target.resolve() or
                    not (member.isfile() or member.isdir())):
                raise RuntimeError('Unsafe or unsupported tar member: ' + member.name)
    verified.append({'archive': name, 'sha256': actual, 'destination': str(target), 'members': len(members)})
for record in verified:
    target = Path(record['destination'])
    target.mkdir(parents=True, exist_ok=False)
    with tarfile.open(root / 'deployment' / record['archive'], 'r:gz') as archive:
        archive.extractall(target)
receipt = {'status': 'INSTALLED', 'hostname': socket.gethostname(),
           'utc': datetime.now(timezone.utc).isoformat(), 'archives': verified}
with (root / 'deployment/install_receipt.json').open('x') as handle:
    json.dump(receipt, handle, indent=2)
    handle.write('\n')
print(json.dumps(receipt))
