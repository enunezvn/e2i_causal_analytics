"""Stress tests for large-scale validation.

These tests are NOT run by default in CI.
Run with: pytest tests/stress/ -m stress

Performance targets:
- GES: <60s on 100K rows
- PC: <120s on 100K rows
- Memory: <8GB for 100K row datasets
"""
