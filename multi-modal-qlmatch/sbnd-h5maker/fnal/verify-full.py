#!/usr/bin/env python3
"""fm4npp #14: check the full run against the list. For each pair: <id>.done present, status OK, every
listed file present with the .done size; collect events / wc_missing / check problems from the json.
Writes lists/status-<sample>.tsv and lists/sha256sum-<sample>.txt; prints a summary.
usage: verify-full.py <list.tsv> <output base> <lists dir>"""
import sys, os, json, collections
lst, base, ld = sys.argv[1:4]
rows = [l.rstrip('\n').split('\t') for l in open(lst) if not l.startswith('#')]
st = collections.defaultdict(list); sha = collections.defaultdict(list); summ = collections.defaultdict(collections.Counter)
for i, s, r1, tar, nev in rows:
    d = f'{base}/{s}'; done = f'{d}/{i}.done'; why = ''; ev = wm = prob = '-'
    if not os.path.exists(done): status = 'NODONE'
    else:
        lines = [l.rstrip('\n').split('\t') for l in open(done)]
        status = 'OK' if lines[0][1].startswith('OK') else 'BAD'
        for l in lines:
            if l[0] != 'file': continue
            size, h, name = int(l[1]), l[2], l[3]
            p = f'{d}/{name}'
            if not os.path.exists(p) or os.path.getsize(p) != size: status = 'SIZE'; why += name + ' '
            sha[s].append(f'{h}  {name}')
        try:
            j = json.load(open(f'{d}/sbnd_{s}_{i}.h5.stats.json')); ev, wm = j['events'], j['wc_missing']
            c = json.load(open(f'{d}/sbnd_{s}_{i}.check.json')); prob = len(c['problems'])
            if prob: why += 'problems:' + ';'.join(map(str, c['problems']))[:200]
        except Exception as e: status = 'JSON'; why += repr(e)[:100]
    st[s].append(f'{i}\t{status}\t{nev}\t{ev}\t{wm}\t{prob}\t{why}')
    summ[s][status] += 1
    if isinstance(ev, int): summ[s]['events'] += ev; summ[s]['wc_missing'] += wm; summ[s]['problems'] += prob
for s in st:
    open(f'{ld}/status-{s}.tsv', 'w').write('#id\tstatus\treco1_events\th5_events\twc_missing\tcheck_problems\twhy\n' + '\n'.join(st[s]) + '\n')
    open(f'{ld}/sha256sum-{s}.txt', 'w').write('\n'.join(sorted(sha[s], key=lambda x: x.split('  ')[1])) + '\n')
    print(s, dict(summ[s]))
