"""Probe subsystem: template parsing, embedding generator, content-addressed store.

A probe is a labeled token (or short phrase) that's been forward-passed
through the model and its embedding recorded at one or more depths. A
ProbeSet is a collection of probe embeddings plus their metadata, generated
from a template file that lists rows/columns/cells of labeled tokens.

The Embedding Generator is a core operation (not a measurement module) —
it consumes user-submitted templates and writes to the ProbeStore. Probe-
using measurements declare a ProbeRequirement referencing a template_id
and capture_signature; the framework looks up the matching ProbeSet in
the store and passes it to compute().
"""
from tagm.probes.artifact import ProbeSet, ProbeEmbedding
from tagm.probes.store import ProbeStore
from tagm.probes.template import ProbeTemplate, load_template, parse_template_csv
from tagm.probes.generator import EmbeddingGenerator

__all__ = [
    "ProbeSet",
    "ProbeEmbedding",
    "ProbeStore",
    "ProbeTemplate",
    "load_template",
    "parse_template_csv",
    "EmbeddingGenerator",
]
