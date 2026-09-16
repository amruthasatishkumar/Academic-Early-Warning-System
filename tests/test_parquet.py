import io
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
        logical = str(pq.ParquetFile(io.BytesIO(payload)).schema.column(0).logical_type).lower()
        self.assertIn("microseconds", logical)
        self.assertNotIn("nanoseconds", logical)
        pd.testing.assert_frame_equal(read_frame(io.BytesIO(payload)), original)

    def test_submicrosecond_values_are_not_silently_truncated(self):
        original = pd.DataFrame({"time": pd.to_datetime(["2026-01-01T00:00:00.123456789Z"], utc=True)})
        with self.assertRaises(pa.ArrowInvalid):
            write_frame(original, io.BytesIO())


if __name__ == "__main__":
    unittest.main()
