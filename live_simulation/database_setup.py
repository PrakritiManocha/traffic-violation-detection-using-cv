import sqlite3

def initialize_database():
    conn = sqlite3.connect("vision_guard.db")
    cursor = conn.cursor()
    
    # Core violations ledger
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_id TEXT UNIQUE,
            timestamp REAL,
            plate_id TEXT,
            violation_class TEXT,
            confidence REAL,
            legal_narrative TEXT,
            cryptographic_hash TEXT
        )
    """)
    
    # Spatial/Traffic analysis metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic_analytics (
            frame_id INTEGER,
            timestamp REAL,
            active_nodes INTEGER,
            edge_latency_ms REAL
        )
    """)
    
    conn.commit()
    conn.close()
    print("[DATABASE] SQLite relational storage engines initialized successfully.")

if __name__ == "__main__":
    initialize_database()