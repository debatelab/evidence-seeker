# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Removed

## [0.1.3b1] - 2025-11-06

### Changed

- Upgrade dependencies: From now one, we are using llama-index-workflows>=2.0.0. (3b4e576419ca49473991490c8d3f11c61ff7f1c4)

## [0.1.2] - 2025-11-04

### Added

+ PostgreSQL Vector Store Support: The retriever now supports PostgreSQL as a vector store backend, enabling more robust and scalable storage solutions for production deployments.
+ Metadata Filtering: Enhanced retrieval capabilities with powerful metadata filtering that allows you to narrow down search results before similarity search.
+ Dynamic Index Updates: Maintain and update your document repository without rebuilding the entire index. Works seamlessly with both file-based and PostgreSQL backends.
+ Asynchronous Operations: All index operations now support asynchronous execution.
+ Progress Tracking: Monitor index building and update operations with customizable callbacks.

## [0.1.0] - 2017-06-20

First public release.



