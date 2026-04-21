"""TAGM instrument layer — adapter, capture, deltas, pipeline, cache.

This layer produces raw captured data from model forward passes and
weight-delta computations. It has no opinion about what measurements
are computed; it exposes stable data structures that the measurement
layer reads from.
"""
