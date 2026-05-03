"""Train/val/test stratified splitter (shard 01 §A).

Emits a 1-D ``stratify`` array (``np.int64``) suitable as the ``y`` arg to
``StratifiedKFold(...).split(X, y)`` per shard 21 §B.

Filled in by commit 03 (consumed by commit 06).
"""
