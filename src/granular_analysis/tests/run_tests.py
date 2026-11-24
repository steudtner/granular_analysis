import sys
import os
import traceback

# Ensure repo root (postproc) is on sys.path so we can import package modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

tests = [
    'tests.test_alpha2d',
]

failed = 0
for t in tests:
    try:
        m = __import__(t, fromlist=['*'])
        # run all functions starting with test_
        for name in dir(m):
            if name.startswith('test_'):
                func = getattr(m, name)
                print(f'Running {t}.{name}...')
                func()
    except Exception:
        failed += 1
        traceback.print_exc()

if failed:
    print(f"{failed} test modules failed")
    sys.exit(2)
else:
    print('All tests passed')
