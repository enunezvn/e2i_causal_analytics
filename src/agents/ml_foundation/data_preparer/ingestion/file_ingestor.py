"""File ingestion strategies for data_preparer.

Domain-agnostic file readers that dispatch on extension. Accepts either a
directory (containing canonical e2i_ml_v3_* files) or an explicit mapping
of logical names to paths. Returns pandas DataFrames verbatim — no cleaning,
no transformation. Downstream schema_validator / quality_checker catch bad
data. Converter scripts are responsible for shaping upstream.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Protocol, Union

import pandas as pd

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised when a file cannot be read or dispatched."""


class Reader(Protocol):
    """Format-specific reader strategy."""

    extensions: tuple[str, ...]

    def read(self, path: Path) -> pd.DataFrame:
        ...


class ParquetReader:
    extensions: tuple[str, ...] = (".parquet", ".pq")

    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)


class JsonReader:
    extensions: tuple[str, ...] = (".json",)

    def read(self, path: Path) -> pd.DataFrame:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise IngestionError(
                f"JSON file {path} must contain a top-level list of records"
            )
        return pd.DataFrame(records)


class CsvReader:
    extensions: tuple[str, ...] = (".csv",)

    def read(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)


# Canonical filenames the ingestion path looks for in a directory.
# Order matters: patient_journeys is the primary target; the rest are
# ancillary and loaded if present.
CANONICAL_FILES: tuple[str, ...] = (
    "e2i_ml_v3_patient_journeys",
    "e2i_ml_v3_treatment_events",
    "e2i_ml_v3_hcp_profiles",
)


class FileIngestor:
    """Dispatch file reads on extension; accept directory or explicit mapping.

    Two calling modes:
      1. ``ingest_directory(Path)``: looks for canonical e2i_ml_v3_* files in
         the directory, trying each registered extension in order.
      2. ``ingest_mapping({"patient_journeys": "/path/to/file.parquet", ...})``:
         reads each file by its path, keying by the mapping's logical name.

    Both return a dict[str, DataFrame] keyed by logical name.
    """

    def __init__(self, readers: list[Reader] | None = None) -> None:
        self.readers: list[Reader] = readers or [
            ParquetReader(),
            JsonReader(),
            CsvReader(),
        ]
        # Pre-build extension → reader index
        self._by_ext: Dict[str, Reader] = {}
        for r in self.readers:
            for ext in r.extensions:
                self._by_ext[ext.lower()] = r

    def _reader_for(self, path: Path) -> Reader:
        ext = path.suffix.lower()
        reader = self._by_ext.get(ext)
        if reader is None:
            raise IngestionError(
                f"No reader registered for extension '{ext}' (path: {path})"
            )
        return reader

    def ingest_file(self, path: Path) -> pd.DataFrame:
        """Read a single file, dispatching on extension."""
        path = Path(path)
        if not path.exists():
            raise IngestionError(f"File not found: {path}")
        reader = self._reader_for(path)
        logger.debug("Reading %s via %s", path, type(reader).__name__)
        return reader.read(path)

    def ingest_directory(self, directory: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """Read canonical e2i_ml_v3_* files from a directory.

        For each canonical stem, tries each registered extension in order
        and loads the first match. Missing ancillary files are skipped with
        a debug log; missing patient_journeys raises.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise IngestionError(f"Not a directory: {directory}")

        result: Dict[str, pd.DataFrame] = {}
        for stem in CANONICAL_FILES:
            logical_name = stem.replace("e2i_ml_v3_", "")
            df = self._try_read_stem(directory, stem)
            if df is not None:
                result[logical_name] = df
            elif logical_name == "patient_journeys":
                raise IngestionError(
                    f"Required file 'e2i_ml_v3_patient_journeys.*' not found in {directory} "
                    f"(tried extensions: {tuple(self._by_ext.keys())})"
                )
            else:
                logger.debug("Optional file '%s.*' not found in %s", stem, directory)
        return result

    def ingest_mapping(
        self, paths: Mapping[str, Union[str, Path]]
    ) -> Dict[str, pd.DataFrame]:
        """Read files from an explicit {logical_name: path} mapping."""
        result: Dict[str, pd.DataFrame] = {}
        for name, path in paths.items():
            result[name] = self.ingest_file(Path(path))
        return result

    def _try_read_stem(self, directory: Path, stem: str) -> pd.DataFrame | None:
        for ext in self._by_ext:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return self.ingest_file(candidate)
        return None
