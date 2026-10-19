"""Fetch the pinned ADS source needed for provenance checks; no ADS weights."""
import io
import json
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = '64f2889b192cee2280ec7bf1405752b81ec02bc1'


def main():
    response = requests.get(f'https://codeload.github.com/active-diffusion-subsampling/ads/zip/{REVISION}', timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for member in archive.infolist():
            relative = Path(*Path(member.filename).parts[1:])
            if member.is_dir() or not relative.parts:
                continue
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive path')
            target = ROOT / 'artifacts/ads_official' / relative
            raw = archive.read(member)
            if target.exists():
                if target.read_bytes() != raw:
                    raise ValueError(f'Existing reference differs: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
    print(json.dumps({'repository': 'https://github.com/active-diffusion-subsampling/ads', 'revision': REVISION}))


if __name__ == '__main__':
    main()
