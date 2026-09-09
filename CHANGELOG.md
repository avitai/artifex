# Changelog

All notable changes to artifex are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- CI runs every pre-commit hook and checks that `uv.lock` is current in the quality
  gate, runs the unit tests and the package build on Python 3.13 as well as 3.12, and
  checks distributions with `twine check --strict`.
- Publishing uses PyPI trusted publishing (OIDC) instead of an API token, with a
  `github-release` dispatch target that creates the GitHub Release for an existing
  tag and then uploads; `RELEASING.md` records the checklist.

## [0.1.4] - 2026-08-29

Releases up to 0.1.4 predate this file; their notes are the
[GitHub Releases](https://github.com/avitai/artifex/releases).
