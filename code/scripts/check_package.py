"""Check local links, media, finite results and the repository-only file scope."""
from pathlib import Path
import hashlib,json,math,re
from urllib.parse import unquote
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
skip={'.git','__pycache__','.pytest_cache','.venv','__MACOSX'}
files=[p for p in ROOT.rglob('*') if p.is_file() and not any(s in p.relative_to(ROOT).parts for s in skip)]
for p in files:
    assert p.suffix.lower() not in {'.tex','.pdf','.csv','.mw'}, f'Excluded file type: {p}'
    assert p.name!='.DS_Store'

links=[]
for p in ROOT.rglob('*.md'):
    text=p.read_text()
    refs=re.findall(r'\]\(([^)]+)\)',text)+re.findall(r'(?:src|href)="([^"]+)"',text)
    for ref in refs:
        if ref.startswith(('http:','https:','mailto:','#')):continue
        target=unquote(ref.split('#',1)[0].strip())
        if not target:continue
        resolved=(p.parent/target).resolve()
        assert resolved.is_relative_to(ROOT),f'Escaping local link: {p}: {ref}'
        assert resolved.exists(),f'Broken local link: {p}: {ref}'
        links.append(ref)

def finite(value):
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,list):return all(finite(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)

for p in (ROOT/'code/results').glob('*.json'):
    assert finite(json.loads(p.read_text())),f'Nonfinite number in {p.name}'

gifs=[]
manifest=json.loads((ROOT/'assets/animations/manifest.json').read_text())
for p in sorted((ROOT/'assets/animations').glob('*.gif')):
    with Image.open(p) as im:
        assert im.n_frames>=4 and im.width>=1400 and im.height>=600
        assert im.info.get('loop')==0
        unique=set();durations=[]
        for i in range(im.n_frames):
            im.seek(i);frame=im.convert('RGB');frame.thumbnail((120,80))
            unique.add(hashlib.sha256(frame.tobytes()).hexdigest())
            durations.append(im.info.get('duration',0))
        assert len(unique)>=4,f'No meaningful motion: {p.name}'
        assert im.n_frames==manifest[p.stem]['frames']
        gifs.append({'file':p.name,'frames':im.n_frames,'distinct_thumbnails':len(unique),
                     'pixels':list(im.size),'loop_duration_ms':sum(durations)})
assert len(gifs)==7

figures=[]
for p in sorted((ROOT/'figures/png').glob('*.png')):
    with Image.open(p) as im:
        assert im.width>=6000 and im.info.get('dpi',(0,))[0]>=999
        if im.mode=='RGBA':assert im.getchannel('A').getextrema()==(255,255)
        assert im.convert('RGB').getpixel((0,0))==(255,255,255)
        figures.append({'file':p.name,'pixels':list(im.size),'dpi':list(im.info['dpi'])})
assert len(figures)==6
assert len(list((ROOT/'figures/svg').glob('*.svg')))==6
report={'status':'passed','local_links_checked':len(links),'animations':gifs,'figures':figures,
        'scope':'No LaTeX, PDF, CSV or old Maple worksheet files','finite_result_values':True}
(ROOT/'verification/media_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'PASS: {len(links)} local links, 7 animated GIFs, 6 high-resolution PNGs, 6 vector SVGs, finite results and file scope.')
