# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Removed

## [0.1.4b] - 2025-11-25

### Changed

+ Error handlung and patience (via `tenacity`) for `HFTextEmbeddingsInference`. [8134435
](https://github.com/debatelab/evidence-seeker/commit/8134435f124196ecac90da7beee75538022c5187)

### Fixed

+ Corrected default values RetrievalConig.

## [0.1.3] - 2025-11-25

### Changed

+ New handling of config defaults: All defaults are loaded from package data. [7f5c588](https://github.com/debatelab/evidence-seeker/commit/7f5c58876bdb8bf950dc81fa02215a8e0ddbc5cb)
+ Improved prompt templates for preprocessing (disambiguation). [b83b535](https://github.com/debatelab/evidence-seeker/commit/b83b535125b2184ea9124dbf6da1ec0a3046adaa)
    + Known issue: [Llama 3.3 70B Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) has problems with the listing step.

### Fixed

+ Fixed incorrect `asyncpg` query. [3b4e576](https://github.com/debatelab/evidence-seeker/commit/3b4e576419ca49473991490c8d3f11c61ff7f1c4)

## [0.1.3b1] - 2025-11-06

+ Beta release for version 0.1.3. (released on PyPI and tagged in GitHub).

### Changed

- Upgrade dependencies: From now one, we are using llama-index-workflows>=2.0.0. [553c2c3](https://github.com/debatelab/evidence-seeker/commit/553c2c30268c3d0a8a0f663fec1584071908533b)

## [0.1.2] - 2025-11-04

### Added

+ PostgreSQL Vector Store Support: The retriever now supports PostgreSQL as a vector store backend, enabling more robust and scalable storage solutions for production deployments.
+ Metadata Filtering: Enhanced retrieval capabilities with powerful metadata filtering that allows you to narrow down search results before similarity search.
+ Dynamic Index Updates: Maintain and update your document repository without rebuilding the entire index. Works seamlessly with both file-based and PostgreSQL backends.
+ Asynchronous Operations: All index operations now support asynchronous execution.
+ Progress Tracking: Monitor index building and update operations with customizable callbacks.

## [0.1.0] - 2017-06-20

First public release.



