"""Score the observable installation history, including withdrawal of the tools."""
import argparse
import json
from pathlib import Path
import numpy as np

from task import TOP, PATCHES


def evaluate(history):
    def centre(row, body):
        return np.asarray(row['centres'][body])

    def clear(row):
        glass = centre(row, 'glass')
        return (centre(row, 'carrier')[2] - glass[2] > .035 and
                all(np.linalg.norm(centre(row, f'liner{i}')-glass) > .085 for i in range(PATCHES)))

    first = history[0]
    lifted = any(r['covered'] >= .99 and centre(r, 'glass')[2] > TOP + .065 for r in history)
    peeling = [r for r in history if .04 < r['covered'] < .96 and centre(r,'glass')[2] > TOP + .04]
    bare = any(r['covered'] == 0 and r['peel_front']==PATCHES and centre(r,'glass')[2]>TOP+.04
               and np.linalg.norm((centre(r,'glass')-centre(r,'phone'))[:2])>.08 for r in history)
    endpoint = history[-20:]
    checks = dict(started_beside_phone=np.linalg.norm((centre(first,'glass')-centre(first,'phone'))[:2])>.12,
                  lifted_with_backing=lifted, partial_peel_observed=len(peeling)>=2,
                  exposed_before_transfer=bare,
                  tools_clear_at_end=all(clear(r) for r in endpoint),
                  settled_installation=len(endpoint)==20 and all(r['success'] for r in endpoint))
    return {k:bool(v) for k,v in checks.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=Path)
    checks = evaluate(json.loads(parser.parse_args().trace.read_text()))
    print(json.dumps({'success': all(checks.values()), 'checks': checks}, indent=2))
