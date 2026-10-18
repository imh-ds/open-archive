"""Demo of open.simple_recursive on simulated data with a known two-level chain.

    py -3.11 examples/simple_recursive_ols_demo.py [N]

True structure: X4 -> X1 -> Y <- X3 <- X2; X5..X10 are noise.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from open.simple_recursive import recursive_ols  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
r = np.random.default_rng(2026)
d = {f'X{i}': r.normal(size=n) for i in (2, 4, 5, 6, 7, 8, 9, 10)}
d['X1'] = .5 * d['X4'] + r.normal(size=n)
d['X3'] = .5 * d['X2'] + r.normal(size=n)
d['Y'] = .3 * d['X1'] + .3 * d['X3'] + r.normal(size=n)

res = recursive_ols(pd.DataFrame(d), 'Y')
print(res.summary())
print('\nAll tests:')
print(res.tests.round(3).to_string(index=False))
