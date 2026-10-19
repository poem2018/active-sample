"""Prepare five actual CelebA test images using bounded ZIP range requests.

The aligned original images are taken from a revision-pinned public mirror.
No full image archive is downloaded. This does not reproduce an unpublished
ADS CelebA preprocessing recipe; the center-crop recipe is explicit below.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
import urllib.request
import zipfile
import zlib

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/celeba128_ads_v1"
REVISION = "35ebb31405e9da028409199a494f2e77eb10f253"
BASE = f"https://huggingface.co/datasets/Yuehao/celeba/resolve/{REVISION}/"
SELECTION_SEED = 2026101701
DOWNLOAD_LIMIT = 5_000_000
ZIP_BYTES = 1443490838
CENTRAL_START = 1421407350
PARTITION_BYTES = 2836386


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


class Downloader:
    def __init__(self):
        self.bytes_received = 0
        self.requests = []

    def get(self, name: str, start: int | None = None,
            end: int | None = None, expected_size: int | None = None) -> bytes:
        url = BASE + name
        headers = {"User-Agent": "active-sample-research/1.0"}
        if start is not None:
            assert end is not None and end >= start
            # Include the range in the URL to avoid intermediary cache confusion.
            url += f"?range={start}_{end}"
            headers["Range"] = f"bytes={start}-{end}"
            expected_size = end - start + 1
        assert expected_size is not None
        if self.bytes_received + expected_size > DOWNLOAD_LIMIT:
            raise RuntimeError("Predeclared 5 MB transfer budget would be exceeded")
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=45) as response:
            if start is not None:
                assert response.status == 206, "Server did not honor HTTP Range"
                content_range = response.headers.get("Content-Range", "")
                assert content_range.startswith(f"bytes {start}-{end}/"), content_range
                if name.endswith(".zip"):
                    assert content_range.endswith(f"/{ZIP_BYTES}"), content_range
            else:
                assert response.status == 200
                content_range = None
            data = response.read(expected_size + 1)
            self.bytes_received += len(data)
            assert len(data) == expected_size, (name, len(data), expected_size)
            self.requests.append(dict(url=url, status=response.status,
                                      content_range=content_range, bytes=len(data)))
        return data


def fetch_original(downloader: Downloader, image_id: int) -> tuple[bytes, dict]:
    # This pinned ZIP has one 99-byte folder entry followed by 109-byte image
    # entries. Every accessed signature and filename is verified independently.
    offset = CENTRAL_START + 99 + (image_id - 1) * 109
    entry = downloader.get("img_align_celeba.zip", offset, offset + 108)
    fields = struct.unpack("<4s6H3L5H2L", entry[:46])
    assert fields[0] == b"PK\x01\x02"
    method, crc32, compressed_size, uncompressed_size = fields[4], fields[7], fields[8], fields[9]
    filename_size, extra_size, comment_size = fields[10:13]
    assert 46 + filename_size + extra_size + comment_size == 109
    filename = entry[46:46 + filename_size].decode()
    assert filename == f"img_align_celeba/{image_id:06d}.jpg", filename
    assert method == 0 and compressed_size == uncompressed_size
    local_offset = fields[-1]
    header = downloader.get("img_align_celeba.zip", local_offset, local_offset + 29)
    local = struct.unpack("<4s5H3L2H", header)
    assert local[0] == b"PK\x03\x04" and local[3] == method
    assert local[6] == crc32 and local[7] == compressed_size
    name_extra = downloader.get("img_align_celeba.zip", local_offset + 30,
                                local_offset + 29 + local[-2] + local[-1])
    assert name_extra[:local[-2]].decode() == filename
    data_offset = local_offset + 30 + local[-2] + local[-1]
    raw = downloader.get("img_align_celeba.zip", data_offset,
                         data_offset + compressed_size - 1)
    assert zlib.crc32(raw) == crc32
    return raw, dict(archive_member=filename, central_directory_offset=offset,
                     local_header_offset=local_offset, image_data_offset=data_offset,
                     compressed_bytes=compressed_size, crc32=f"{crc32:08x}",
                     sha256=sha(raw), original_source_url=BASE + "img_align_celeba.zip")


def preprocess(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert("RGB")
        assert image.size == (178, 218), image.size
        image = image.crop((0, 20, 178, 198)).resize((128, 128), Image.Resampling.BILINEAR)
        return np.asarray(image).transpose(2, 0, 1).astype(np.float32) / np.float32(127.5) - np.float32(1)


def validate_cache(manifest: dict) -> None:
    assert manifest["selection_seed"] == SELECTION_SEED and len(manifest["cases"]) == 5
    assert manifest["revision"] == REVISION
    assert manifest["preparation_script_sha256"] == sha(Path(__file__).read_bytes())
    partition = (OUT / "list_eval_partition.txt").read_bytes()
    assert sha(partition) == manifest["partition_sha256"]
    partition_by_id = {int(name.split(".")[0]): int(p)
                       for name, p in (line.split() for line in partition.decode().splitlines())}
    assert len(set(manifest["image_ids"])) == 5
    for case in manifest["cases"]:
        raw = (ROOT / case["original_path"]).read_bytes()
        truth_path = ROOT / case["truth_path"]
        assert sha(raw) == case["original_sha256"]
        assert zlib.crc32(raw) == int(case["archive"]["crc32"], 16)
        assert sha(truth_path.read_bytes()) == case["truth_sha256"]
        truth = np.load(truth_path, allow_pickle=False)
        assert truth.shape == (3, 128, 128) and truth.dtype == np.float32
        assert np.array_equal(truth, preprocess(raw))
        assert partition_by_id[case["image_id"]] == 2




def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare the fixed five-image roster; never reselect test images.")
    parser.add_argument('--image-dir', type=Path, help="Existing img_align_celeba directory (no image download).")
    parser.add_argument('--partition-file', type=Path, help="Existing official list_eval_partition.txt.")
    args = parser.parse_args()
    reference = json.loads((ROOT / 'configs/dataset_reference.json').read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        validate_cache(manifest)
        assert manifest['image_ids'] == reference['image_ids']
        for case, expected in zip(manifest['cases'], reference['cases']):
            assert case['original_sha256'] == expected['original_sha256']
            assert case['truth_sha256'] == expected['truth_sha256']
        print(json.dumps(dict(status='validated_existing_cache', image_ids=manifest['image_ids'])))
        return
    downloader = Downloader()
    partition_path = OUT / 'list_eval_partition.txt'
    if args.partition_file:
        partition = args.partition_file.read_bytes()
    elif partition_path.exists():
        partition = partition_path.read_bytes()
    else:
        partition = downloader.get('list_eval_partition.txt', expected_size=PARTITION_BYTES)
    assert sha(partition) == reference['partition_sha256'], "Official partition does not match reference"
    partition_path.write_bytes(partition)
    partitions = {int(name[:6]): int(split) for name, split in
                  (line.split() for line in partition.decode().splitlines())}
    selection = dict(image_ids=reference['image_ids'], selection_seed=SELECTION_SEED,
                     partition_sha256=sha(partition),
                     rule='Replay the previously locked project test roster; do not select new images.')
    save_json(OUT / 'selection.json', selection)
    cases = []
    for expected in reference['cases']:
        iid = expected['image_id']
        assert partitions[iid] == 2
        if args.image_dir:
            raw = (args.image_dir / f'{iid:06d}.jpg').read_bytes()
            archive = expected['archive']
        else:
            raw, archive = fetch_original(downloader, iid)
        assert sha(raw) == expected['original_sha256'], f'Original image differs: {iid}'
        truth = preprocess(raw)
        folder = OUT / f'image_{iid}'
        folder.mkdir(exist_ok=True)
        original_path = folder / f'{iid:06d}.jpg'
        truth_path = folder / 'truth.npy'
        original_path.write_bytes(raw)
        np.save(truth_path, truth, allow_pickle=False)
        assert sha(truth_path.read_bytes()) == expected['truth_sha256'], f'Preprocessing differs: {iid}'
        cases.append(dict(expected, original_path=str(original_path.relative_to(ROOT)),
                          truth_path=str(truth_path.relative_to(ROOT)), archive=archive))
    manifest = dict(dataset='CelebA original aligned RGB test images',
                    source='https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html',
                    mirror='https://huggingface.co/datasets/Yuehao/celeba', revision=REVISION,
                    partition_sha256=sha(partition), selection_seed=SELECTION_SEED,
                    image_ids=reference['image_ids'], cases=cases, selection=selection,
                    preprocessing=reference['preprocessing'], crop_xyxy=reference['crop_xyxy'],
                    pillow_version=Image.__version__, numpy_version=np.__version__,
                    preparation_script=str(Path(__file__).relative_to(ROOT)),
                    preparation_script_sha256=sha(Path(__file__).read_bytes()),
                    download_bytes=downloader.bytes_received, http_requests=downloader.requests,
                    split_scope='Replay of the five locked official-test images; prior-training overlap is unknown.')
    save_json(manifest_path, manifest)
    validate_cache(manifest)
    print(json.dumps(dict(status='complete', image_ids=manifest['image_ids'],
                         downloaded_bytes=downloader.bytes_received, manifest=str(manifest_path))))


if __name__ == '__main__':
    main()
