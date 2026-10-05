"""VJP rules, one module per op family. Importing this package registers them."""

from . import elementwise, indexing, linalg, nn, reduction, shape  # noqa: F401
