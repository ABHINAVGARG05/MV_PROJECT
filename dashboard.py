import argparse
import csv
import os
from backend.database import WeldDatabase


def print_dashboard(db: WeldDatabase):
    stats   = db.get_stats()
    records = db.get_all()

    total = stats["total"]
    good  = stats["good_count"]
    bad   = stats["bad_count"]
    rate  = (good / total * 100) if total > 0 else 0

    print("\n" + "═"*60)
    print("  WELDING INSPECTION ANALYTICS DASHBOARD")
    print("═"*60)
    print(f"  Total inspected : {total}")
    print(f"  Good welds      : {good}  ({rate:.1f}%)")
    print(f"  Bad welds       : {bad}  ({100 - rate:.1f}%)")
    print()

    if stats["defect_breakdown"]:
        print("  Defect Breakdown:")
        for defect, count in sorted(stats["defect_breakdown"].items(),
                                    key=lambda x: -x[1]):
            bar = "█" * int(count / max(stats["defect_breakdown"].values()) * 20)
            print(f"    {defect:<20} {count:>4}  {bar}")
    print()

    if records:
        print("  Recent Inspections (last 10):")
        print(f"  {'ID':>4}  {'Status':<6}  {'Defect':<18}  Timestamp")
        print("  " + "─"*54)
        for r in records[:10]:
            defect = r["defect_type"] or "─"
            print(f"  {r['id']:>4}  {r['status']:<6}  {defect:<18}  {r['timestamp']}")
    print("═"*60 + "\n")


def export_csv(db: WeldDatabase, path: str):
    records = db.get_all()
    if not records:
        print("No records to export.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"Exported {len(records)} records to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Welding Inspection Dashboard")
    parser.add_argument("--db",     default="weld_results.db", help="Database path")
    parser.add_argument("--export", metavar="CSV_PATH",        help="Export to CSV")
    args = parser.parse_args()

    db = WeldDatabase(args.db)
    print_dashboard(db)

    if args.export:
        export_csv(db, args.export)