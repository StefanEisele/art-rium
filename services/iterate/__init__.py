"""Iterate — the render → critic → editor loop for Z-Image Turbo.

The design, and why it is not the obvious "describe, rewrite, repeat" chain,
is Part 1 of prompts/iterate.md; the material the models get is Part 2 of the
same file. This package is the program that file describes:

  spec.py    reads Part 2 (blocks, role prompts, templates, anchors)
  checks.py  the code-side rules every prompt core must pass before a render
  judge.py   scoring, the accept decision, phases, the plateau jolt
  roles.py   the four model calls (brief, critic, duel, editor) + validation
  runner.py  the background job: GPU hand-over, renders, ingest, series, sidecar
"""
