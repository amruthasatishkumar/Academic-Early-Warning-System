import base64
import io
import json
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from aews.parquet import read_frame, write_frame


class ParquetTimestampTests(unittest.TestCase):
    def test_microsecond_physical_type_and_lossless_roundtrip(self):
        original = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01T00:00:00.123456Z", None], utc=True),
            "value": [1, 2],
        })
        stream = io.BytesIO()
        write_frame(original, stream)
        payload = stream.getvalue()
        parquet = pq.ParquetFile(io.BytesIO(payload))
        logical = str(parquet.schema.column(0).logical_type).lower()
        self.assertIn("microseconds", logical)
        self.assertNotIn("nanoseconds", logical)
        embedded = pa.ipc.read_schema(pa.BufferReader(base64.b64decode(parquet.metadata.metadata[b"ARROW:schema"])))
        self.assertEqual(embedded.field("time").type, pa.timestamp("us", tz="UTC"))
        pandas_metadata = json.loads(parquet.metadata.metadata[b"pandas"])
        self.assertEqual(pandas_metadata["columns"][0]["numpy_type"], "datetime64[us]")
        self.assertEqual(parquet.metadata.format_version, "1.0")
        pd.testing.assert_frame_equal(read_frame(io.BytesIO(payload)), original)

    def test_naive_and_non_utc_timestamps_preserve_values_without_mutating_input(self):
        original = pd.DataFrame({
            "naive": pd.to_datetime(["1969-12-31T23:59:59.123456", None]),
            "local": pd.to_datetime(["2026-01-01T00:00:00.123456Z", None], utc=True).tz_convert("America/Los_Angeles"),
            "category": pd.Categorical(["a", "b"]),
            "nullable": pd.array([1, None], dtype="Int64"),
        }, index=[4, 7])
        before = original.copy(deep=True)
        stream = io.BytesIO()
        write_frame(original, stream)
        pd.testing.assert_frame_equal(original, before)
        pd.testing.assert_frame_equal(read_frame(io.BytesIO(stream.getvalue())), original.reset_index(drop=True))

    def test_submicrosecond_values_are_not_silently_truncated(self):
        original = pd.DataFrame({"time": pd.to_datetime(["2026-01-01T00:00:00.123456789Z"], utc=True)})
        with self.assertRaises(pa.ArrowInvalid):
            write_frame(original, io.BytesIO())


if __name__ == "__main__":
    unittest.main()
