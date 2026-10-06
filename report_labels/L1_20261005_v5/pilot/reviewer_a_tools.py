import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from report_label_extraction.l1 import CATEGORIES, AUDIT_DIMENSIONS, compact

p = Path(__file__).resolve().parents[1]
samples = list(csv.DictReader((p / 'pilot_sample.csv').open(encoding='utf-8-sig')))[:60]
cache_path = p / 'pilot/extraction_cache.jsonl'
cache = {r['report_hash']: r['result'] for r in map(json.loads, cache_path.open(encoding='utf-8'))} if cache_path.exists() else {}
out = p / 'pilot/review_a.csv'
seen = {r['report_hash'] for r in csv.DictReader(out.open(encoding='utf-8'))} if out.exists() else set()
fields = ['StudyInstanceUID', 'category', 'report_hash', 'report_status', 'evidence', 'meets_criteria', 'audit_reasons', 'disposition'] + AUDIT_DIMENSIONS + ['reviewer', 'notes']
if len(sys.argv) == 1:
    source = {r['report_hash']: r['report'] for r in map(json.loads, (p / 'reports_source.jsonl').open(encoding='utf-8'))}
    pending = [(i, r) for i, r in enumerate(samples) if r['report_hash'] in cache and r['report_hash'] not in seen]
    print('CACHE', len(cache), 'DONE', len(seen), 'PENDING', [i for i, _ in pending])
    if pending:
        i, r = pending[0]
        x = cache[r['report_hash']]
        print('INDEX', i, 'REPORT', source[r['report_hash']])
        print('PARSED', compact(x['parsed']))
        print('AUDITS', compact({k: {a: v for a, v in b.items() if a != 'evidence_locations'} for k, b in x['audits'].items()}))
    else:
        print((p / 'pilot/run.json').read_text(encoding='utf-8'))
else:
    decisions = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    with out.open('a', encoding='utf-8', newline='\n') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n')
        if f.tell() == 0:
            w.writeheader()
        for index, entries in decisions.items():
            r = samples[int(index)]
            assert r['report_hash'] not in seen
            assert len(entries) == 12
            x = cache[r['report_hash']]
            for cat, entry in zip(CATEGORIES, entries):
                l, a = x['parsed']['labels'][cat], x['audits'][cat]
                dims = {k: 'PASS' for k in AUDIT_DIMENSIONS}
                dims.update(entry.get('dims', {}))
                disposition = entry.get('disposition', 'KEEP')
                assert not (disposition == 'KEEP' and (a['critical'] or any(v != 'PASS' for v in dims.values())))
                w.writerow({'StudyInstanceUID': r['StudyInstanceUID'], 'category': cat, 'report_hash': r['report_hash'], 'report_status': l['report_status'], 'evidence': compact(l['evidence']), 'meets_criteria': a['meets_criteria'], 'audit_reasons': compact(a['review_reason']), 'disposition': disposition, **dims, 'reviewer': 'GPT-6.1 Sol independent text reviewer A', 'notes': entry['notes']})
            f.flush()
            os.fsync(f.fileno())
