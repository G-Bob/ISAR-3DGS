"""Prepare the frozen BSAT time study and launch explicitly requested groups."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
REFERENCE = PROJECT / 'outputs/bsat_restart_20261004_102638'
STUDY = PROJECT / 'outputs/bsat_observation_days_20261004'
EXPECTED = {1: (73, 59), 3: (248, 199), 7: (695, 556), 13: (1790, 1432)}


def write_once(path, text):
    if path.exists():
        if path.read_text().splitlines() != text.splitlines():
            raise RuntimeError(f'Refusing to change existing artifact: {path}')
    else:
        with path.open('x') as stream:
            stream.write(text)


def utc(text):
    return dt.datetime.fromisoformat(text.replace('Z', '+00:00'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', nargs='*', choices=['D01', 'D03', 'D07', 'D13'], default=[])
    args = parser.parse_args()
    assert len(args.launch) == len(set(args.launch)), 'Duplicate launch group'
    launch = json.loads((REFERENCE / 'launch.json').read_text())
    dataset = Path(launch['dataset'])
    train = (REFERENCE / 'train_cpis.txt').read_text().splitlines()
    heldout = (REFERENCE / 'heldout_cpis.txt').read_text().splitlines()
    assert len(train) == 1903 and len(heldout) == 475
    assert not set(train) & set(heldout)
    names = sorted(train + heldout)
    assert len(set(names)) == 2378
    rows = []
    for name in names:
        folder = dataset / name
        metadata = json.loads((folder / '_COMPLETE.json').read_text())
        start, end = utc(metadata['cpi_start_utc']), utc(metadata['cpi_end_utc'])
        assert start <= end
        assert (folder / 'view_angle.npy').is_file()
        assert len(list((folder / 'isar_images_noaxis').glob('*.png'))) == 1
        rows.append((name, start, end, name in set(train)))
    rows.sort(key=lambda row: (row[1], row[0]))
    t0 = min(row[1] for row in rows)
    STUDY.mkdir(exist_ok=True)
    config_text = (REFERENCE / 'training/run_config.json').read_text()
    write_once(STUDY / 'reference_config.json', config_text)
    write_once(STUDY / 'heldout_cpis.txt', '\n'.join(heldout) + '\n')
    commands = {}
    previous = set()
    summary = []
    for days, expected in EXPECTED.items():
        group = f'D{days:02d}'
        cutoff = t0 + dt.timedelta(days=days)
        available = [row for row in rows if row[2] <= cutoff]
        selected = [row[0] for row in available if row[3]]
        assert (len(available), len(selected)) == expected, (group, len(available), len(selected))
        assert previous <= set(selected) and not set(selected) & set(heldout)
        previous = set(selected)
        folder = STUDY / group
        folder.mkdir(exist_ok=True)
        write_once(folder / 'train_cpis.txt', '\n'.join(selected) + '\n')
        command = list(launch['command'])
        for flag, value in {'--config': STUDY / 'reference_config.json', '--out_dir': folder / 'training', '--encounter_names_file': folder / 'train_cpis.txt'}.items():
            command[command.index(flag) + 1] = str(value)
        record = {'group': group, 'days': days, 't0_utc': t0.isoformat(), 'cutoff_utc': cutoff.isoformat(), 'available_count': len(available), 'train_count': len(selected), 'common_heldout_count': len(heldout), 'command': command, 'threads': 2}
        write_once(folder / 'launch.json', json.dumps(record, indent=2) + '\n')
        summary.append(record)
        commands[group] = command
        print('VALIDATED', group, len(available), len(selected), flush=True)
    write_once(STUDY / 'study_manifest.json', json.dumps({'reference_run': str(REFERENCE), 'groups': summary, 'ALL_train_count': len(train), 'ALL_path': str(REFERENCE)}, indent=2) + '\n')
    import io
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(['cpi', 'start_utc', 'end_utc', 'split', 'D01', 'D03', 'D07', 'D13'])
    for name, start, end, is_train in rows:
        writer.writerow([name, start.isoformat(), end.isoformat(), 'train' if is_train else 'heldout', *[int(end <= t0 + dt.timedelta(days=day)) for day in EXPECTED]])
    write_once(STUDY / 'cpi_membership.csv', csv_buffer.getvalue())
    for group in args.launch:
        folder = STUDY / group
        marker = folder / 'started.json'
        if marker.exists():
            raise RuntimeError(f'{group} already submitted; inspect started.json, do not duplicate')
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    for group in args.launch:
        folder = STUDY / group
        with (folder / 'train.log').open('x') as logfile:
            process = subprocess.Popen(commands[group], cwd=PROJECT, env=environment, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        (folder / 'started.json').write_text(json.dumps({'pid': process.pid, 'submitted_utc': dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2) + '\n')
        print('STARTED', group, 'PID', process.pid, flush=True)


if __name__ == '__main__':
    main()
