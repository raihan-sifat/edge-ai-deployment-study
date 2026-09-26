"""Edge AI Deployment Study -- reproducible CPU benchmarking for efficient models.

The package is organised around four concerns:

``edgebench.data``      deterministic dataset construction
``edgebench.models``    model zoo plus CIFAR-10 stem adaptation
``edgebench.optim``     the deployment optimization ladder
``edgebench.bench``     the measurement protocol
``edgebench.reporting`` charts and tables derived purely from stored results

Every measurement produced by this package is written to disk as JSON first.
Charts and tables are then derived from those files, never recomputed, so a
figure can always be traced back to the run that produced it.
"""

from __future__ import annotations

__all__ = ["RESULTS_SCHEMA_VERSION", "__version__"]

__version__ = "0.1.0"

#: Bumped whenever the on-disk result schema changes in a breaking way.
RESULTS_SCHEMA_VERSION = 1
