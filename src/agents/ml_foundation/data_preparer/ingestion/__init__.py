"""File-based ingestion package for data_preparer.

Provides a domain-agnostic FileIngestor that dispatches on file extension
(parquet / json / csv) and returns canonical DataFrames for downstream
validation nodes. Domain logic (cohort construction, feature engineering,
leakage-safe windowing) lives upstream in the converter scripts; this
package only reads files.
"""

from .file_ingestor import (
    CsvReader,
    FileIngestor,
    IngestionError,
    JsonReader,
    ParquetReader,
    Reader,
)

__all__ = [
    "CsvReader",
    "FileIngestor",
    "IngestionError",
    "JsonReader",
    "ParquetReader",
    "Reader",
]
