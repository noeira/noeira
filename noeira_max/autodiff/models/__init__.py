"""Models written as functions of a parameter dict, over ``max.graph.ops``.

The parameters are a plain ``dict[str, TensorValue]``, so ``value_and_grad``
returns gradients with the same keys. Weights are stored ``[in, out]``
(``x @ w``), the layout the torch twins in ``tests/parity_torch.py`` mirror.
"""
