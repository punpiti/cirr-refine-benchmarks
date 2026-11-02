"""Exercise interrupted writes at every byte, including quoted multiline errors."""
import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from resume_io import recover_csv


class RecoveryTests(unittest.TestCase):
    def test_every_interruption_preserves_only_complete_records(self):
        fields = ["method", "status", "error"]
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        records = [fields, ["ASEM", "completed", ""], ["MIA", "error", 'line one\nline "two"\nผิดพลาด']]
        ends = []
        for record in records:
            writer.writerow(record)
            ends.append(len(stream.getvalue().encode()))
        raw = stream.getvalue().encode()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "results.csv"
            for cutoff in range(len(raw) + 1):
                path.write_bytes(raw[:cutoff])
                recover_csv(path, fields)
                expected = max([end for end in ends if end <= cutoff], default=0)
                actual = path.read_bytes() if path.exists() else b""
                self.assertEqual(actual, raw[:expected], f"interrupted at byte {cutoff}")

    def test_corruption_before_last_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            raw = b"a,b\nonly-one-field\nlast,row\n"
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                recover_csv(path, ["a", "b"])
            self.assertEqual(path.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
