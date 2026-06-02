"""
report.py
─────────
Generate and view attendance reports from the database.

Usage
─────
python report.py                          # today's report (print)
python report.py --date 2025-08-10        # specific date
python report.py --export                 # export today → CSV
python report.py --date 2025-08-10 --export --out my_report.csv
python report.py --range 2025-08-01 2025-08-31  # date-range summary
"""

import argparse
import sqlite3
import pandas as pd
from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from face_attendance.face_attendance import AttendanceDB, DB_FILE, REPORTS_DIR


def report_for_date(db: AttendanceDB, target_date: str) -> pd.DataFrame:
    with db._conn() as con:
        df = pd.read_sql_query(
            """SELECT s.student_id, s.name, a.time_in, a.status
               FROM students s
               LEFT JOIN attendance a
                      ON s.student_id = a.student_id AND a.date = ?
               ORDER BY s.name""",
            con,
            params=(target_date,),
        )
    df["status"] = df["status"].fillna("Absent")
    df["date"]   = target_date
    return df


def report_range(db: AttendanceDB, start: str, end: str) -> pd.DataFrame:
    with db._conn() as con:
        df = pd.read_sql_query(
            """SELECT s.student_id, s.name, a.date, a.time_in, a.status
               FROM students s
               LEFT JOIN attendance a
                      ON s.student_id = a.student_id
               WHERE a.date BETWEEN ? AND ?
               ORDER BY a.date, s.name""",
            con,
            params=(start, end),
        )
    return df


def print_report(df: pd.DataFrame, title: str = "ATTENDANCE REPORT"):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)
    print(df.to_string(index=False))
    if "status" in df.columns:
        present = (df["status"] == "Present").sum()
        total   = len(df)
        print(f"\n  Present: {present}/{total}  ({present / max(total, 1) * 100:.1f}%)")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Attendance report generator")
    parser.add_argument("--date",   default=date.today().isoformat(),
                        help="Date (YYYY-MM-DD). Default: today")
    parser.add_argument("--range",  nargs=2, metavar=("START", "END"),
                        help="Date range (YYYY-MM-DD YYYY-MM-DD)")
    parser.add_argument("--export", action="store_true", help="Export to CSV")
    parser.add_argument("--out",    default=None, help="Output CSV path")
    args = parser.parse_args()

    db = AttendanceDB()

    if args.range:
        df    = report_range(db, args.range[0], args.range[1])
        title = f"RANGE REPORT  {args.range[0]} → {args.range[1]}"
    else:
        df    = report_for_date(db, args.date)
        title = f"ATTENDANCE REPORT  —  {args.date}"

    print_report(df, title)

    if args.export:
        out = Path(args.out) if args.out else REPORTS_DIR / f"report_{args.date}.csv"
        df.to_csv(out, index=False)
        print(f"  Exported → {out}\n")


if __name__ == "__main__":
    main()
