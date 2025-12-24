# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2025-12-24

### Fixed
- Fixed signature class to variant type mapping in MAF file processing
- SBS, DBS, and RNA-SBS signature classes now correctly map to SNP, DBP variant types
- Resolved "'NoneType' object is not subscriptable" error when using signature_class="SBS"

## [0.1.0] - 2025-12-24

### Added
- Initial release
- Core functionality for mutation rate estimation
- Selection coefficient inference
- Multi-signature support
