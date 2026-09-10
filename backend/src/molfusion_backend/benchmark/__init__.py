"""Phase 6A: the frozen molecular representation benchmark protocol.

A protocol, plus the smallest code that proves the protocol is wireable.
This package does not run the benchmark: the full matrix of endpoints,
representations, probes and splits is a later phase, and Phase 6A exists so
that when it runs, none of its decisions are made ad hoc.

    from molfusion_backend.benchmark import protocol
    protocol.protocol_summary()

The prose companion is `docs/benchmark-protocol.md`, which is kept in step
with `protocol.py` by tests.
"""

from molfusion_backend.benchmark import (
    cache,
    datasets,
    evaluate,
    features,
    metrics,
    pipelines,
    protocol,
    results,
    splits,
)

__all__ = [
    "cache",
    "datasets",
    "evaluate",
    "features",
    "metrics",
    "pipelines",
    "protocol",
    "results",
    "splits",
]
