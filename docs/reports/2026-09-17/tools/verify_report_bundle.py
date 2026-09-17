"""Verify report links, archived source hashes, videos, and portable packaging."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import pdfplumber
from pypdf import PdfReader

sys.stdout.reconfigure(encoding='utf-8')
OUT=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[3]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links=[]
    def handle_starttag(self, tag, attrs):
        for key,value in attrs:
            if key in ('href','src'):
                self.links.append(value)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(reviewed):
    pdf=next(OUT.glob('*REV6.pdf'))
    report=json.loads((OUT/'report_integrity.json').read_text(encoding='utf-8'))
    blocks=json.loads((OUT/'report_pages.json').read_text(encoding='utf-8'))
    parser=Links()
    parser.feed((OUT/'START_보고서와영상.html').read_text(encoding='utf-8'))
    external=[p for p in parser.links if not p.startswith('#')]
    assert all((OUT/p).is_file() for p in external)
    reader=PdfReader(pdf)
    assert len(reader.pages)==len(blocks)==51
    bad=[]
    with pdfplumber.open(pdf) as doc:
        for i,page in enumerate(doc.pages,1):
            for char in page.chars:
                if char['x0'] < 35 or char['x1'] > page.width-35 or char['top'] < 28 or char['bottom'] > page.height-12:
                    bad.append((i,char['text'],char['x0'],char['x1'],char['top'],char['bottom']))
    assert not bad,bad[:15]
    source=json.loads((OUT/'code_excerpt_index.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'07_원본판정/단일문_r40/source_manifest.json').read_text(encoding='utf-8'))
    matched=[]
    for key,item in source['sources'].items():
        if '접촉복사본' in item['copy']:
            rel='src/'+item['path'].replace('\\','/').split('/src/',1)[1]
            assert item['sha256']==manifest['inputs_sha256'][rel],rel
            matched.append(rel)
    ffmpeg=ROOT/'tmp/media_tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    videos={p for p in external if p.endswith('.mp4')}
    decoded=[]
    for rel in sorted(videos):
        p=OUT/rel
        proc=subprocess.run([str(ffmpeg),'-hide_banner','-v','error','-i',str(p),'-f','null','-'],
                            capture_output=True,timeout=180)
        assert proc.returncode==0,(rel,proc.stderr[-1000:])
        assert not proc.stderr,(rel,proc.stderr[-1000:])
        probe=subprocess.run([str(ffmpeg),'-hide_banner','-i',str(p)],capture_output=True,
                             text=True,encoding='utf-8',errors='replace')
        assert 'Video: h264' in probe.stderr,rel
        decoded.append({'file':rel,'h264':True,'full_decode_pass':True,'sha256':digest(p)})
    report.update({'visual_qa':'all 51 rendered pages reviewed; selected dense pages inspected individually' if reviewed else 'pending',
                   'text_page_boundary_pass':True,'html_links_checked':len(external),
                   'r40_source_manifest_matches':matched,'videos_full_decode':decoded,
                   'pdf_sha256':digest(pdf)})
    (OUT/'report_integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('videos_full_decode','r40_source_manifest_matches')},ensure_ascii=False,indent=2))
    if reviewed:
        archive=OUT.parent/'팀원공유_핵심아이디어_보고서_REV6_20260917.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for path in sorted(OUT.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    z.write(path,Path('FireRobot_REV6')/path.relative_to(OUT))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            assert 'FireRobot_REV6/START_보고서와영상.html' in z.namelist()
        print(json.dumps({'archive':str(archive),'size_mb':round(archive.stat().st_size/1024**2,2),
                          'zip_integrity_pass':True},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--visual-reviewed',action='store_true')
    main(parser.parse_args().visual_reviewed)
