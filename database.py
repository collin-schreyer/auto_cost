"""
database.py – persistence layer for booth + rack tracking
---------------------------------------------------------
Updated to support enhanced hanging detection with:
- Operation types (manual_hanging, forklift_hanging, etc.)
- LLM explanations for rack events
- Forklift and manual detection flags
- Global rack tracking across zones
- Position-based rack identity management
"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional

# ── DB location ──────────────────────────────────────────────────────────────
DB_FILE = Path(__file__).with_name("booth_monitor.db").as_posix()


# ── Low-level helpers ────────────────────────────────────────────────────────
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ── Schema bootstrap (runs on import) ────────────────────────────────────────
def _init_schema() -> None:
    conn = _connect()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS booth_status (
            region           TEXT PRIMARY KEY,
            in_use           INTEGER,
            entered_at       TEXT,
            last_seen        TEXT,
            door_closed      INTEGER,
            person_detected  INTEGER,
            time_spent_today REAL DEFAULT 0,
            last_updated     TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS booth_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT,
            region          TEXT,
            state           TEXT,           -- 'OCCUPIED' / 'EMPTY'
            duration        REAL,           -- sec (only for EMPTY)
            door_closed     INTEGER,
            person_detected INTEGER
        )
        """
    )

    # Enhanced rack_positions table with global rack ID
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rack_positions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            camera_id   TEXT,
            rack_id     TEXT,                    -- local rack ID (rack_01, rack_02)
            global_rack_id TEXT,                 -- global rack ID across zones
            x           REAL,
            y           REAL,
            moving      INTEGER,
            zone_description TEXT,               -- "left_side", "center", "right_wall", etc.
            confidence  REAL DEFAULT 0.8
        )
        """
    )

    # Enhanced rack_events table with global rack reference
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rack_events (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT,
            camera_id        TEXT,
            rack_id          TEXT,
            global_rack_id   TEXT,              -- reference to global rack
            people_count     INTEGER DEFAULT 0,
            action           TEXT,
            confidence       REAL,
            zone             TEXT,
            operation_type   TEXT,              -- 'manual_hanging', 'forklift_hanging', 'manual_removing', 'forklift_removing', 'other'
            explanation      TEXT,              -- LLM's detailed explanation of what's happening
            forklift_detected INTEGER DEFAULT 0, -- 1 if forklift detected in frames
            manual_detected   INTEGER DEFAULT 0  -- 1 if manual operation detected
        )
        """
    )

    # NEW: Global rack registry table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS global_racks (
            rack_id          TEXT PRIMARY KEY,   -- global unique ID (global_rack_01, global_rack_02, etc.)
            current_camera   TEXT,               -- "general_labor" or "powder_booth" or "in_transit"
            current_zone     TEXT,               -- "left_side", "center", "right_wall", etc.
            position_description TEXT,           -- "near left wall", "center of floor", etc.
            visual_notes     TEXT,               -- any distinguishing features if available
            created_timestamp TEXT,
            last_updated     TEXT,
            last_movement    TEXT,               -- timestamp of last zone change
            status           TEXT DEFAULT 'active'  -- 'active', 'missing', 'in_transit'
        )
        """
    )

    # NEW: Rack movement history table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS rack_movements (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            global_rack_id   TEXT,
            from_camera      TEXT,
            to_camera        TEXT,
            from_zone        TEXT,
            to_zone          TEXT,
            movement_timestamp TEXT,
            detection_method TEXT,               -- "temporal_correlation", "manual_confirmation", etc.
            confidence       REAL DEFAULT 0.8,
            notes           TEXT
        )
        """
    )

    # NEW: Zone definitions table (for position-based tracking)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS zone_definitions (
            camera_id        TEXT,
            zone_name        TEXT,
            description      TEXT,
            x_min           REAL,
            x_max           REAL,
            y_min           REAL,
            y_max           REAL,
            is_transition_zone INTEGER DEFAULT 0,  -- zones where racks might appear in multiple cameras
            PRIMARY KEY (camera_id, zone_name)
        )
        """
    )

    # Check if we need to migrate existing tables
    _migrate_existing_tables(cur)

    conn.commit()
    conn.close()


def _migrate_existing_tables(cur) -> None:
    """Handle migrations for existing installations"""
    
    # Check rack_positions table structure
    cur.execute("PRAGMA table_info(rack_positions)")
    rack_pos_columns = [row[1] for row in cur.fetchall()]
    
    if 'global_rack_id' not in rack_pos_columns:
        print("Adding global_rack_id to rack_positions table...")
        new_columns = [
            ("global_rack_id", "TEXT"),
            ("zone_description", "TEXT"),
            ("confidence", "REAL DEFAULT 0.8")
        ]
        
        for column_name, column_type in new_columns:
            try:
                cur.execute(f"ALTER TABLE rack_positions ADD COLUMN {column_name} {column_type}")
                print(f"Added column to rack_positions: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"Column {column_name} already exists in rack_positions, skipping...")
                else:
                    raise

    # Check rack_events table structure
    cur.execute("PRAGMA table_info(rack_events)")
    rack_events_columns = [row[1] for row in cur.fetchall()]
    
    # Migration from person_id to people_count (existing migration)
    if 'person_id' in rack_events_columns and 'people_count' not in rack_events_columns:
        print("Migrating rack_events table from person_id to people_count...")
        # Create new table with updated schema
        cur.execute(
            """
            CREATE TABLE rack_events_new (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT,
                camera_id        TEXT,
                rack_id          TEXT,
                global_rack_id   TEXT,
                people_count     INTEGER DEFAULT 0,
                action           TEXT,
                confidence       REAL,
                zone             TEXT,
                operation_type   TEXT,
                explanation      TEXT,
                forklift_detected INTEGER DEFAULT 0,
                manual_detected   INTEGER DEFAULT 0
            )
            """
        )
        
        # Copy existing data, converting person_id presence to people_count=1
        cur.execute(
            """
            INSERT INTO rack_events_new 
            (timestamp, camera_id, rack_id, people_count, action, confidence, zone, 
             operation_type, explanation, forklift_detected, manual_detected)
            SELECT timestamp, camera_id, rack_id, 
                   CASE WHEN person_id IS NOT NULL AND person_id != '' THEN 1 ELSE 0 END,
                   action, confidence, zone,
                   'legacy', 'Migrated from old schema', 0, 
                   CASE WHEN person_id IS NOT NULL AND person_id != '' THEN 1 ELSE 0 END
            FROM rack_events
            """
        )
        
        # Replace old table
        cur.execute("DROP TABLE rack_events")
        cur.execute("ALTER TABLE rack_events_new RENAME TO rack_events")
        print("Migration from person_id completed.")
    
    # Migration to add global_rack_id to existing rack_events
    elif 'people_count' in rack_events_columns and 'global_rack_id' not in rack_events_columns:
        print("Adding global_rack_id to rack_events table...")
        try:
            cur.execute("ALTER TABLE rack_events ADD COLUMN global_rack_id TEXT")
            print("Added global_rack_id column to rack_events")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e).lower():
                print("global_rack_id already exists in rack_events, skipping...")
            else:
                raise
    
    # Migration to add enhanced detection fields (for existing people_count tables)
    if 'people_count' in rack_events_columns and 'operation_type' not in rack_events_columns:
        print("Adding enhanced detection fields to rack_events table...")
        
        # Add new columns to existing table
        new_columns = [
            ("operation_type", "TEXT"),
            ("explanation", "TEXT"), 
            ("forklift_detected", "INTEGER DEFAULT 0"),
            ("manual_detected", "INTEGER DEFAULT 0")
        ]
        
        for column_name, column_type in new_columns:
            try:
                cur.execute(f"ALTER TABLE rack_events ADD COLUMN {column_name} {column_type}")
                print(f"Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    print(f"Column {column_name} already exists, skipping...")
                else:
                    raise
        
        # Update existing records with default values
        cur.execute("""
            UPDATE rack_events 
            SET operation_type = 'legacy',
                explanation = 'Pre-enhancement record',
                forklift_detected = 0,
                manual_detected = CASE WHEN people_count > 0 THEN 1 ELSE 0 END
            WHERE operation_type IS NULL
        """)
        print("Enhanced fields migration completed.")


# ── Booth helpers (unchanged) ────────────────────────────────────────────────────────────
def update_booth_status(
    region: str,
    in_use: bool,
    door_closed: Optional[bool] = None,
    person_detected: Optional[bool] = None,
    now_iso: Optional[str] = None,
) -> None:
    """
    Upsert current booth state.
    """
    now_iso = now_iso or _now_iso()
    conn = _connect()
    conn.execute(
        """
        INSERT INTO booth_status
              (region,in_use,entered_at,last_seen,
               door_closed,person_detected,last_updated)
        VALUES (?,?,?,?,?,?,?)
        ON CONFLICT(region) DO UPDATE SET
            in_use          = excluded.in_use,
            entered_at      = excluded.entered_at,
            last_seen       = excluded.last_seen,
            door_closed     = excluded.door_closed,
            person_detected = excluded.person_detected,
            last_updated    = excluded.last_updated
        """,
        (
            region,
            int(in_use),
            now_iso if in_use else None,
            now_iso,
            int(door_closed) if door_closed is not None else None,
            int(person_detected) if person_detected is not None else None,
            now_iso,
        ),
    )
    conn.commit()
    conn.close()


def add_booth_history(
    region: str,
    state: str,
    *,
    duration_seconds: Optional[float] = None,
    door_closed: Optional[bool] = None,
    person_detected: Optional[bool] = None,
    when: Optional[str] = None,
) -> None:
    """
    Insert an immutable OCCUPIED / EMPTY record.
    """
    when = when or _now_iso()
    conn = _connect()
    conn.execute(
        """
        INSERT INTO booth_history
              (timestamp,region,state,duration,door_closed,person_detected)
        VALUES (?,?,?,?,?,?)
        """,
        (
            when,
            region,
            state,
            duration_seconds,
            int(door_closed) if door_closed is not None else None,
            int(person_detected) if person_detected is not None else None,
        ),
    )
    conn.commit()
    conn.close()


def update_time_spent(region: str, seconds: float) -> None:
    """
    Increment today's time_spent_today counter.
    """
    conn = _connect()
    conn.execute(
        """
        UPDATE booth_status
           SET time_spent_today = COALESCE(time_spent_today,0) + ?,
               last_updated     = ?
         WHERE region = ?
        """,
        (seconds, _now_iso(), region),
    )
    conn.commit()
    conn.close()


# ── Enhanced Rack helpers with global tracking ─────────────────────────────────────────────────────────────
def add_rack_positions(positions: List[Dict]) -> None:
    """
    Bulk-insert centroid + motion rows (one per rack per frame).
    Enhanced to support global rack IDs and zone descriptions.
    """
    if not positions:
        return
    conn = _connect()
    conn.executemany(
        """
        INSERT INTO rack_positions
              (timestamp,camera_id,rack_id,global_rack_id,x,y,moving,zone_description,confidence)
        VALUES (:timestamp,:camera_id,:rack_id,:global_rack_id,:x,:y,:moving,:zone_description,:confidence)
        """,
        positions,
    )
    conn.commit()
    conn.close()


def add_rack_events(events: List[Dict]) -> None:
    """
    ULTRA-SAFE version that handles any missing fields gracefully
    """
    if not events:
        return
    
    import logging
    logger = logging.getLogger(__name__)
    
    conn = _connect()
    
    # Get the actual table schema to match exactly
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(rack_events)")
    columns_info = cur.fetchall()
    
    # Get column names (excluding 'id' which is auto-increment)
    column_names = [col[1] for col in columns_info if col[1] != 'id']
    logger.debug(f"Database columns: {column_names}")
    
    processed_events = []
    for event in events:
        # Create a new event dict with only the columns that exist in the database
        processed_event = {}
        
        for col_name in column_names:
            if col_name in event:
                processed_event[col_name] = event[col_name]
            else:
                # Provide safe defaults
                if col_name == "timestamp":
                    processed_event[col_name] = _now_iso()
                elif col_name in ["camera_id", "rack_id", "global_rack_id"]:
                    processed_event[col_name] = event.get(col_name, None)
                elif col_name in ["people_count", "forklift_detected", "manual_detected"]:
                    processed_event[col_name] = int(event.get(col_name, 0))
                elif col_name == "confidence":
                    processed_event[col_name] = float(event.get(col_name, 0.0))
                elif col_name in ["action", "operation_type", "zone"]:
                    processed_event[col_name] = event.get(col_name, "unknown")
                elif col_name == "explanation":
                    processed_event[col_name] = event.get(col_name, "No explanation provided")
                else:
                    processed_event[col_name] = None
        
        processed_events.append(processed_event)
    
    # Build the SQL dynamically based on actual columns
    placeholders = ", ".join([f":{col}" for col in column_names])
    columns_str = ", ".join(column_names)
    
    sql = f"INSERT INTO rack_events ({columns_str}) VALUES ({placeholders})"
    logger.debug(f"Generated SQL: {sql}")
    logger.debug(f"Sample data: {processed_events[0] if processed_events else 'None'}")
    
    try:
        conn.executemany(sql, processed_events)
        conn.commit()
        logger.info(f"✅ Successfully stored {len(processed_events)} events")
        
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        logger.error(f"SQL: {sql}")
        logger.error(f"Data: {processed_events[0] if processed_events else 'None'}")
        raise
    finally:
        conn.close()


# ── Global Rack Management Functions ─────────────────────────────────────────────────────────────
def get_global_rack_state(camera_id: Optional[str] = None) -> List[Dict]:
    """
    Get current global rack state, optionally filtered by camera.
    """
    conn = _connect()
    cur = conn.cursor()
    
    if camera_id:
        cur.execute("""
            SELECT * FROM global_racks 
            WHERE current_camera = ? OR current_camera = 'in_transit'
            ORDER BY rack_id
        """, (camera_id,))
    else:
        cur.execute("SELECT * FROM global_racks ORDER BY rack_id")
    
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def update_global_rack_position(rack_id: str, camera_id: str, zone: str, 
                               position_description: str = "", 
                               visual_notes: str = "") -> None:
    """
    Update the current position of a global rack.
    """
    now = _now_iso()
    conn = _connect()
    
    # Check if this is a movement (different camera)
    cur = conn.cursor()
    cur.execute("SELECT current_camera, current_zone FROM global_racks WHERE rack_id = ?", (rack_id,))
    existing = cur.fetchone()
    
    if existing:
        old_camera, old_zone = existing
        if old_camera != camera_id:
            # This is a movement - log it
            add_rack_movement(rack_id, old_camera, camera_id, old_zone, zone)
    
    # Update the rack position
    conn.execute(
        """
        INSERT INTO global_racks
              (rack_id, current_camera, current_zone, position_description, 
               visual_notes, created_timestamp, last_updated, last_movement, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
        ON CONFLICT(rack_id) DO UPDATE SET
            current_camera = excluded.current_camera,
            current_zone = excluded.current_zone,
            position_description = excluded.position_description,
            visual_notes = CASE WHEN excluded.visual_notes != '' 
                              THEN excluded.visual_notes 
                              ELSE visual_notes END,
            last_updated = excluded.last_updated,
            last_movement = CASE WHEN current_camera != excluded.current_camera 
                               THEN excluded.last_movement 
                               ELSE last_movement END,
            status = 'active'
        """,
        (rack_id, camera_id, zone, position_description, visual_notes, now, now, now)
    )
    
    conn.commit()
    conn.close()


def add_rack_movement(global_rack_id: str, from_camera: str, to_camera: str,
                     from_zone: str, to_zone: str, 
                     detection_method: str = "temporal_correlation",
                     confidence: float = 0.8, notes: str = "") -> None:
    """
    Log a rack movement between zones.
    """
    conn = _connect()
    conn.execute(
        """
        INSERT INTO rack_movements
              (global_rack_id, from_camera, to_camera, from_zone, to_zone,
               movement_timestamp, detection_method, confidence, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (global_rack_id, from_camera, to_camera, from_zone, to_zone,
         _now_iso(), detection_method, confidence, notes)
    )
    conn.commit()
    conn.close()


def get_recent_movements(hours: int = 24) -> List[Dict]:
    """
    Get recent rack movements within the specified time window.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM rack_movements 
        WHERE movement_timestamp > datetime('now', '-{} hours')
        ORDER BY movement_timestamp DESC
    """.format(hours))
    
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def mark_rack_missing(rack_id: str) -> None:
    """
    Mark a rack as missing when it hasn't been seen for a while.
    """
    conn = _connect()
    conn.execute("""
        UPDATE global_racks 
        SET status = 'missing', 
            current_camera = NULL,
            current_zone = NULL,
            last_updated = ?
        WHERE rack_id = ?
    """, (_now_iso(), rack_id))
    conn.commit()
    conn.close()


def get_racks_in_camera(camera_id: str) -> List[Dict]:
    """
    Get all racks currently in a specific camera zone.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM global_racks 
        WHERE current_camera = ? AND status = 'active'
        ORDER BY current_zone, rack_id
    """, (camera_id,))
    
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── Zone Management Functions ─────────────────────────────────────────────────────────────
def add_zone_definition(camera_id: str, zone_name: str, description: str,
                       x_min: float = None, x_max: float = None,
                       y_min: float = None, y_max: float = None,
                       is_transition_zone: bool = False) -> None:
    """
    Define a zone within a camera view for position-based tracking.
    """
    conn = _connect()
    conn.execute(
        """
        INSERT INTO zone_definitions
              (camera_id, zone_name, description, x_min, x_max, y_min, y_max, is_transition_zone)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(camera_id, zone_name) DO UPDATE SET
            description = excluded.description,
            x_min = excluded.x_min,
            x_max = excluded.x_max,
            y_min = excluded.y_min,
            y_max = excluded.y_max,
            is_transition_zone = excluded.is_transition_zone
        """,
        (camera_id, zone_name, description, x_min, x_max, y_min, y_max, int(is_transition_zone))
    )
    conn.commit()
    conn.close()


def get_zone_definitions(camera_id: str) -> List[Dict]:
    """
    Get zone definitions for a camera.
    """
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM zone_definitions WHERE camera_id = ?", (camera_id,))
    
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── Enhanced queries (existing functions updated) ─────────────────────────────────────────────────────────────
def get_all_booth_status() -> List[Dict]:
    conn = _connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM booth_status")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_booth_history(region: Optional[str] = None, limit: int = 50) -> List[Dict]:
    conn = _connect()
    cur = conn.cursor()
    sql = "SELECT * FROM booth_history"
    params: List = []
    if region:
        sql += " WHERE region = ?"
        params.append(region)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    cur.execute(sql, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_rack_events_by_type(operation_type: Optional[str] = None, 
                           camera_id: Optional[str] = None,
                           limit: int = 50) -> List[Dict]:
    """
    Get rack events filtered by operation type and/or camera.
    """
    conn = _connect()
    cur = conn.cursor()
    
    sql = "SELECT * FROM rack_events WHERE 1=1"
    params: List = []
    
    if operation_type:
        sql += " AND operation_type = ?"
        params.append(operation_type)
    
    if camera_id:
        sql += " AND camera_id = ?"
        params.append(camera_id)
    
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    cur.execute(sql, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_operation_statistics(hours: int = 24) -> Dict:
    """
    Get statistics on operation types over the last N hours.
    """
    conn = _connect()
    cur = conn.cursor()
    
    # Count by operation type
    cur.execute("""
        SELECT operation_type, COUNT(*) as count
        FROM rack_events 
        WHERE timestamp > datetime('now', '-{} hours')
        AND operation_type IS NOT NULL
        GROUP BY operation_type
    """.format(hours))
    
    operation_counts = {row[0]: row[1] for row in cur.fetchall()}
    
    # Count manual vs forklift
    cur.execute("""
        SELECT 
            SUM(manual_detected) as manual_operations,
            SUM(forklift_detected) as forklift_operations,
            COUNT(*) as total_operations
        FROM rack_events 
        WHERE timestamp > datetime('now', '-{} hours')
    """.format(hours))
    
    detection_stats = dict(cur.fetchone())
    
    conn.close()
    
    return {
        "operation_counts": operation_counts,
        "detection_stats": detection_stats,
        "hours_analyzed": hours
    }


def get_events_with_explanations(camera_id: Optional[str] = None, 
                                limit: int = 20) -> List[Dict]:
    """
    Get recent rack events that have explanations, for review purposes.
    """
    conn = _connect()
    cur = conn.cursor()
    
    sql = """
        SELECT * FROM rack_events 
        WHERE explanation IS NOT NULL 
        AND explanation != ''
        AND explanation != 'Migrated from old schema'
    """
    params: List = []
    
    if camera_id:
        sql += " AND camera_id = ?"
        params.append(camera_id)
    
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    
    cur.execute(sql, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


# ── Debugging and maintenance helpers ────────────────────────────────────────
def get_table_info() -> Dict:
    """
    Get information about the current database schema for debugging.
    """
    conn = _connect()
    cur = conn.cursor()
    
    # Get table info
    cur.execute("PRAGMA table_info(rack_events)")
    rack_events_columns = [{"name": row[1], "type": row[2]} for row in cur.fetchall()]
    
    cur.execute("PRAGMA table_info(global_racks)")
    global_racks_columns = [{"name": row[1], "type": row[2]} for row in cur.fetchall()]
    
    # Get record counts
    cur.execute("SELECT COUNT(*) FROM rack_events")
    total_events = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM rack_events WHERE operation_type IS NOT NULL")
    enhanced_events = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM global_racks")
    global_racks_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM rack_movements")
    movements_count = cur.fetchone()[0]
    
    conn.close()
    
    return {
        "rack_events_columns": rack_events_columns,
        "global_racks_columns": global_racks_columns,
        "total_events": total_events,
        "enhanced_events": enhanced_events,
        "legacy_events": total_events - enhanced_events,
        "global_racks_count": global_racks_count,
        "movements_count": movements_count
    }


# ── public init for app.py ───────────────────────────────────────────────────
def init_database() -> None:
    """
    External entry point: initialize database schema with migrations.
    """
    _init_schema()
    
    # Print migration status for debugging
    info = get_table_info()
    print(f"Database initialized: {info['total_events']} total events, "
          f"{info['enhanced_events']} with enhanced detection data, "
          f"{info['global_racks_count']} global racks tracked, "
          f"{info['movements_count']} movements logged")