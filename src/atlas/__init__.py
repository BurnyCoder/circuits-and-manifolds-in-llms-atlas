"""Atlas of Circuits & Manifolds — data pipeline.

Global context: each module in ``atlas.entries`` recomputes one mechanistic-interpretability finding on an open
model (or trains a tiny model) and exports a compact JSON file for the website (``site/public/data/<slug>.json``)
plus raw metrics and a replication verdict under ``results/<slug>/``. ``atlas.run`` is the single entry point.
"""
