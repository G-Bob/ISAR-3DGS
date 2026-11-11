import numpy as np
def read_ply(path):
    f = open(path, 'rb'); names = []; fmt = None; n = 0; types = []
    while True:
        l = f.readline().decode().strip()
        if l.startswith('format'): fmt = l.split()[1]
        elif l.startswith('element vertex'): n = int(l.split()[2])
        elif l.startswith('property'): types.append(l.split()[1]); names.append(l.split()[2])
        elif l == 'end_header': break
    tmap = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'uchar': 'u1', 'int': 'i4'}
    if fmt == 'ascii':
        a = np.loadtxt(f, max_rows=n); return {k: a[:, i] for i, k in enumerate(names)}
    dt = np.dtype([(k, ('<' if 'little' in fmt else '>') + tmap[t]) for k, t in zip(names, types)])
    a = np.fromfile(f, dtype=dt, count=n); return {k: a[k] for k in names}
