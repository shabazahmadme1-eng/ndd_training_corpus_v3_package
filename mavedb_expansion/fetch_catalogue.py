"""Download the MaveDB score-set catalogue (metadata only) with retries; resumable."""
import json, time, urllib.request, urllib.error
from pathlib import Path
OUT = Path(__file__).with_name('mavedb_all_scoresets.json')
URL = 'https://api.mavedb.org/api/v1/score-sets/search'
PAGE = 50

def post(body, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), method='POST',
                                         headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
            return json.load(urllib.request.urlopen(req, timeout=180))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            wait = 5*(k+1)
            print(f'  retry {k+1} after {type(e).__name__} {getattr(e, "code", "")}; sleeping {wait}s', flush=True)
            time.sleep(wait)
    raise RuntimeError(f'gave up at offset {body["offset"]}')

sets = json.loads(OUT.read_text()) if OUT.exists() else []
total = None
while total is None or len(sets) < total:
    r = post({'limit': PAGE, 'offset': len(sets)})
    total = r['numScoreSets']
    if not r['scoreSets']:
        break
    sets += r['scoreSets']
    OUT.write_text(json.dumps(sets))
    print(f'{len(sets)}/{total}', flush=True)
print('done', len(sets), 'unique', len({s["urn"] for s in sets}))
