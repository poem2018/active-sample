"""Fetch pinned official code and weights; never overwrite an existing artifact."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import requests
from .common import ROOT, sha256, write_json
from .official_face_prior import UPSTREAM, CHECKPOINT, CHECKPOINT_SHA256

REVISION='5a1fd4dd3ea7db764ce370a84ce35f9c8b15fde6'
ZIP_SHA256='070dc68c7f903802f6f6e9db0a9ef9f3f7c61668b69dcdb0c2af4d1e3a6eade8'


def main():
    response=requests.get(f'https://codeload.github.com/gnobitab/RectifiedFlow/zip/{REVISION}',timeout=120)
    response.raise_for_status();assert hashlib.sha256(response.content).hexdigest()==ZIP_SHA256
    archive=zipfile.ZipFile(io.BytesIO(response.content));count=0
    for member in archive.infolist():
        relative=Path(*Path(member.filename).parts[1:])
        if member.is_dir() or not relative.parts:continue
        if relative.parts[0]!='ImageGeneration' and str(relative)!='README.md':continue
        assert '..' not in relative.parts and not relative.is_absolute()
        target=UPSTREAM/relative;raw=archive.read(member)
        if target.exists():assert target.read_bytes()==raw,target
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        count+=1
    write_json(UPSTREAM/'upstream.json',{'repository':'https://github.com/gnobitab/RectifiedFlow',
               'commit':REVISION,'archive_sha256':ZIP_SHA256,'files':count})
    if CHECKPOINT.exists():
        assert sha256(CHECKPOINT)==CHECKPOINT_SHA256
        print('Existing official checkpoint verified');return
    CHECKPOINT.parent.mkdir(parents=True,exist_ok=True)
    response=requests.get('https://drive.usercontent.google.com/download',
                          params={'id':'1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF','export':'download','confirm':'t'},
                          stream=True,timeout=120)
    response.raise_for_status()
    if 'text/html' in response.headers.get('Content-Type',''):
        raise RuntimeError('Download returned HTML; inspect official download page')
    temporary=CHECKPOINT.with_suffix('.download')
    with temporary.open('wb') as f:
        for chunk in response.iter_content(8*1024*1024):f.write(chunk)
    assert sha256(temporary)==CHECKPOINT_SHA256
    temporary.replace(CHECKPOINT)
    print('Downloaded and verified',CHECKPOINT)


if __name__=='__main__':main()
