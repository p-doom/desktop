"""Test package.

Present so the suite's own helpers are importable as ``tests.support.*`` under a
bare ``pytest`` invocation and not only under ``python -m pytest``, which is the
difference between the suite running in CI and the suite running on a laptop.
"""
