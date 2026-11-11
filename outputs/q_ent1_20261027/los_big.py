"""Large LOS-error robustness (2026-10-28): sigma_LOS = 5 and 10 deg for SC-GS (lambda_ent = 1, outputs/q_ent1_20261027/L7, L8)
and the three Gaussian baselines (outputs/baseline_robust_20261026/<method>_los5p0, _los10p0). Same machinery and
claim-by-mkdir queues as ent1_suite.py and baseline_suite.py. usage: python los_big.py N_PROPOSED_WORKERS N_BASELINE_WORKERS"""
import sys, threading
sys.path.insert(0, 'outputs/q_ent1_20261027'); sys.path.insert(0, 'outputs/baseline_robust_20261026')
import ent1_suite as es  # patches main_suite: LAM = 1.0, GEN_PERTURB from run['perturb']
import baseline_suite as bs
ms = es.ms
ms.RUNS = [es.P(ms.R('L7_sh3_mask_los5p0', 1, 0, 0, 'sh3'), 'los:5.0'), es.P(ms.R('L8_sh3_mask_los10p0', 1, 0, 0, 'sh3'), 'los:10.0')]
bs.RUNS = [(f'{m}_{c}', m, spec) for c, spec in (('los5p0', 'los:5.0'), ('los10p0', 'los:10.0')) for m in bs.METHODS]
if __name__ == '__main__':
    n1, n2 = (int(a) for a in sys.argv[1:3])
    ts = [threading.Thread(target=ms.loop, args=(i,)) for i in range(n1)] + [threading.Thread(target=bs.loop, args=(i,)) for i in range(n2)]
    [t.start() for t in ts]; [t.join() for t in ts]; print('QUEUE_EMPTY', flush=True)
