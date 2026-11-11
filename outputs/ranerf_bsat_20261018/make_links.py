"""Build a starlink-style link dataset of the frozen 1903 training CPIs for the ISAR-NeRF/RaNeRF framework."""
import os, sys
from pathlib import Path
SRC = Path('/rds/homes/g/gey/Yao/programs/GEIST-6/datasets/bsat')
names = Path(sys.argv[1]).read_text().split()
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=False)
for k, n in enumerate(names, 1):
    d = out / ('starlink%04d' % k); (d / 'Es').mkdir(parents=True)
    os.symlink(SRC / n / 'view_angle.npy', d / 'Es' / 'view_angle.npy')
    os.symlink(SRC / n / 'isar_images_noaxis', d / 'isar_images_noaxis')
(out / 'cpi_order.txt').write_text(chr(10).join(names) + chr(10))
print('built', len(names), out)
