"""Validate the actual published archive, not regenerated test fixtures."""
import collections
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def test_published_archive_matches_manifest_and_protocol():
    base = ROOT / 'data/reproduction'
    manifest = json.loads((base / 'manifest.json').read_text())
    protocol = json.loads((ROOT / 'configs/r03_nb2_review_protocol.json').read_text())
    archive = base / manifest['archive']['path']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive']['sha256']
    expected_counts = {'development': 6008, 'confirmation': 2605, 'external': 1268}
    all_scene_ids = set()
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {f['path'] for f in manifest['files']}
        for item in manifest['files']:
            path = PurePosixPath(item['path'])
            assert not path.is_absolute() and '..' not in path.parts
            data = z.read(item['path'])
            assert len(data) == item['bytes']
            assert hashlib.sha256(data).hexdigest() == item['sha256']
        for name, record in protocol['datasets'].items():
            data = z.read(record['relative_path'])
            assert hashlib.sha256(data).hexdigest() == record['sha256']
            rows = [json.loads(line) for line in data.splitlines()]
            assert len(rows) == expected_counts[name]
            ids = {row['scene_id'] for row in rows}
            assert len(ids) == len(rows)
            assert not all_scene_ids.intersection(ids)
            all_scene_ids.update(ids)
            assert {row['airport_id'] for row in rows} == {record['airport']}
            counts = dict(collections.Counter(row['split'] for row in rows))
            assert counts == manifest['datasets'][name]['splits']
            if name == 'development':
                assert counts == {'train': 3318, 'validation': 1247, 'test': 1443}
