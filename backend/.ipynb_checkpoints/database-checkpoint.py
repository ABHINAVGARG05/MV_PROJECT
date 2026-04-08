import sqlite3
import json
from datetime import datetime
from typing import Optional

class WeldDatabase:

    SCHEMA_IMAGES = """
    CREATE TABLE IF NOT EXISTS weld_images (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        image_path  TEXT    NOT NULL,
        status      TEXT    NOT NULL,      -- 'Good' | 'Bad'
        defect_type TEXT,                  -- NULL for Good welds
        confidence  REAL,
        timestamp   TEXT    NOT NULL
    );
    """

    SCHEMA_FEATURES = """
    CREATE TABLE IF NOT EXISTS extracted_features (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id        INTEGER NOT NULL REFERENCES weld_images(id),
        contrast        REAL,
        energy          REAL,
        homogeneity     REAL,
        correlation     REAL,
        dissimilarity   REAL,
        asm             REAL,
        lbp_energy      REAL,
        lbp_entropy     REAL,
        lbp_mean        REAL,
        lbp_std         REAL,
        area            INTEGER,
        perimeter       REAL,
        aspect_ratio    REAL,
        circularity     REAL,
        edge_density    REAL,
        extent          REAL,
        solidity        REAL,
        equiv_diameter  REAL,
        mean_intensity  REAL,
        std_intensity   REAL,
        skewness        REAL,
        kurtosis        REAL,
        extra_json      TEXT
    );
    """

    def __init__(self, db_path: str = "weld_results.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(self.SCHEMA_IMAGES)
            conn.execute(self.SCHEMA_FEATURES)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save(
        self,
        image_path:  str,
        status:      str,
        defect_type: Optional[str],
        features:    dict,
        confidence:  float = 0.0,
    ) -> int:
        ts = datetime.now().isoformat(timespec="seconds")
        known_keys = {
            "contrast", "energy", "homogeneity", "correlation", "dissimilarity", "asm",
            "lbp_energy", "lbp_entropy", "lbp_mean", "lbp_std",
            "area", "perimeter", "aspect_ratio", "circularity", "edge_density",
            "extent", "solidity", "equiv_diameter",
            "mean_intensity", "std_intensity", "skewness", "kurtosis",
        }
        extra = {k: v for k, v in features.items() if k not in known_keys}

        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO weld_images (image_path, status, defect_type, confidence, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (image_path, status, defect_type, confidence, ts),
            )
            image_id = cur.lastrowid

            conn.execute(
                f"""INSERT INTO extracted_features
                    (image_id, {", ".join(known_keys)}, extra_json)
                    VALUES (?, {", ".join(["?"]*len(known_keys))}, ?)""",
                (
                    image_id,
                    *[features.get(k) for k in sorted(known_keys)],
                    json.dumps(extra) if extra else None,
                ),
            )
        return image_id

    def get_all(self) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM weld_images ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM weld_images").fetchone()[0]
            good  = conn.execute(
                "SELECT COUNT(*) FROM weld_images WHERE status='Good'"
            ).fetchone()[0]
            bad   = total - good
            rows  = conn.execute(
                """SELECT defect_type, COUNT(*) as cnt
                   FROM weld_images WHERE status='Bad'
                   GROUP BY defect_type"""
            ).fetchall()
        return {
            "total"            : total,
            "good_count"       : good,
            "bad_count"        : bad,
            "defect_breakdown" : {r[0]: r[1] for r in rows},
        }

    def get_features_for_image(self, image_id: int) -> Optional[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM extracted_features WHERE image_id=?", (image_id,)
            ).fetchone()
        return dict(row) if row else None