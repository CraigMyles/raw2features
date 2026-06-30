"""Packaged JSON Schemas for the embeddings-store format.

One schema per format version: ``embeddings_store-<version>.schema.json`` (the version
is the store header's ``schema_version``). The schema is the normative definition of the
header; ``raw2features.spec.validate_store`` loads and applies it. Shipped inside the
package so it travels with the wheel and any installed copy can validate offline.
"""
