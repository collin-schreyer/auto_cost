"""
app.py – booth + rack monitor (LM-Studio vision) with Enhanced Dashboard & Image Review
-------------------------------------------------------------------------------------
Phase 2 Updates: Enhanced action detection with forklift support and LLM explanations
- Current status images (latest photo per region)
- 4-frame bursts for rack movement verification  
- 4-frame bursts for hanging parts events
- Enhanced operation type detection (manual vs forklift)
- Detailed LLM explanations for debugging and review
"""

import os, json, time, shutil, logging, re
from datetime import datetime, timezone, timedelta
from threading import Thread
from typing import Dict, List
from collections import deque
from pathlib import Path

from flask import Flask, render_template, jsonify, send_file, abort, request, redirect, url_for

# ── local helpers ────────────────────────────────────────────────────────────
from drive_utils import (
    authenticate_drive,
    download_latest_images,
    delete_files,
    REGIONS,
)
from lm_detection     import detect_person_in_booth
from action_detection import detect_rack_actions  # Enhanced version
from prompt_examples import validate_examples_setup  # Import from correct module
from rack_detection import detect_racks, detect_cross_zone_movements, validate_rack_detection_setup, get_rack_detection_summary             
from database         import (
    init_database,
    update_booth_status,
    add_booth_history,
    update_time_spent,
    get_all_booth_status,
    get_booth_history,
    add_rack_events,
    add_rack_positions,                                
    _connect,
    get_rack_events_by_type,
    get_operation_statistics,
    get_events_with_explanations,
)

# ── config ───────────────────────────────────────────────────────────────────
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))    # seconds
FRAME_SEQUENCE = int(os.getenv("FRAME_SEQUENCE", "4"))     # images / burst
EVIDENCE_DIR   = os.getenv("EVIDENCE_DIR", "evidence_hanging")

# ── Deletion control ───────────────────────────────────────────────────────
ENABLE_DELETION = os.getenv("ENABLE_DELETION", "false").lower() == "true"

# ── Enhanced detection settings ─────────────────────────────────────────────
USE_ENHANCED_DETECTION = os.getenv("USE_ENHANCED_DETECTION", "true").lower() == "true"

# ── Image review directories ────────────────────────────────────────────────
REVIEW_BASE_DIR = "review_images"
CURRENT_IMAGES_DIR = os.path.join(REVIEW_BASE_DIR, "current_status")
RACK_BURSTS_DIR = os.path.join(REVIEW_BASE_DIR, "rack_bursts")
HANGING_BURSTS_DIR = os.path.join(REVIEW_BASE_DIR, "hanging_bursts")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("powder_monitor")

# ── init --------------------------------------------------------------------
init_database()
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(CURRENT_IMAGES_DIR, exist_ok=True)
os.makedirs(RACK_BURSTS_DIR, exist_ok=True)
os.makedirs(HANGING_BURSTS_DIR, exist_ok=True)

# Create subdirectories for each region
for region in REGIONS:
    os.makedirs(os.path.join(RACK_BURSTS_DIR, region), exist_ok=True)
    os.makedirs(os.path.join(HANGING_BURSTS_DIR, region), exist_ok=True)

booth_state: Dict[str, Dict] = {r: {"in_use": False, "entered_at": None} for r in REGIONS}
ACTION_BUFFERS: Dict[str, deque] = {r: deque(maxlen=FRAME_SEQUENCE) for r in REGIONS}
RACK_BUFFERS:   Dict[str, deque] = {r: deque(maxlen=FRAME_SEQUENCE) for r in REGIONS}

KEEP_FILE_IDS: set[str] = set()   # evidence frames
DELETE_QUEUE:  List[str] = []     # Drive IDs to delete each cycle

drive = authenticate_drive()
app   = Flask(__name__)

# ── Enhanced detection validation ────────────────────────────────────────────
def validate_enhanced_detection():
    """Validate that enhanced detection is properly configured"""
    try:
        validation = validate_examples_setup()
        
        if validation["examples_configured"]:
            logger.info("✅ Enhanced detection: %d examples available (%s)", 
                       len(validation["available_examples"]), 
                       ', '.join(validation["available_examples"]))
        else:
            logger.warning("⚠️ Enhanced detection: No examples configured - using basic prompts")
            logger.info("💡 To improve accuracy, add example images to: example_frames/")
            logger.info("   Required: manual_hanging_1-4.png, forklift_hanging_1-4.png, etc.")
        
        return validation["examples_configured"]
        
    except Exception as exc:
        logger.error("❌ Enhanced detection validation failed: %s", exc)
        return False
    
def validate_rack_detection():
    """Validate that rack detection is properly configured"""
    try:
        validation = validate_rack_detection_setup()
        
        if validation["setup_status"] == "configured":
            logger.info("✅ Rack detection: %s", validation["examples_status"])
        else:
            logger.warning("⚠️ Rack detection: %s", validation["examples_status"])
            logger.info("💡 To improve rack tracking, add example images to: rack_examples/")
        
        total_issues = validation.get("total_issues", 0)
        if total_issues > 0:
            logger.warning(f"🔍 Rack tracking has {total_issues} issues - check /api/rack-detection-status")
        
        return validation["setup_status"] == "configured"
        
    except Exception as exc:
        logger.error("❌ Rack detection validation failed: %s", exc)
        return False

# ── Image saving helpers ─────────────────────────────────────────────────────
def save_current_status_image(region: str, image_path: str) -> None:
    """Save the latest image for current status review"""
    try:
        dst_path = os.path.join(CURRENT_IMAGES_DIR, f"{region}_current.png")
        shutil.copy2(image_path, dst_path)
        logger.debug("📷 Saved current status image: %s → %s", image_path, dst_path)
    except Exception as exc:
        logger.warning("Failed to save current status image for %s: %s", region, exc)

def save_rack_burst(region: str, burst_frames: List[Dict], timestamp: str) -> None:
    """Save 4-frame burst for rack movement verification"""
    try:
        # Create timestamp-based folder
        safe_timestamp = timestamp.replace(":", "-").replace("T", "_").replace("Z", "")
        burst_dir = os.path.join(RACK_BURSTS_DIR, region, safe_timestamp)
        os.makedirs(burst_dir, exist_ok=True)
        
        for i, frame in enumerate(burst_frames, 1):
            src_path = frame["path"]
            dst_path = os.path.join(burst_dir, f"frame_{i}.png")
            shutil.copy2(src_path, dst_path)
            
        logger.debug("📸 Saved rack burst for %s: %s (%d frames)", region, burst_dir, len(burst_frames))
    except Exception as exc:
        logger.warning("Failed to save rack burst for %s: %s", region, exc)

def save_hanging_burst(region: str, burst_frames: List[Dict], timestamp: str) -> None:
    """Save 4-frame burst for hanging parts verification"""
    try:
        # Create timestamp-based folder
        safe_timestamp = timestamp.replace(":", "-").replace("T", "_").replace("Z", "")
        burst_dir = os.path.join(HANGING_BURSTS_DIR, region, safe_timestamp)
        os.makedirs(burst_dir, exist_ok=True)
        
        for i, frame in enumerate(burst_frames, 1):
            src_path = frame["path"]
            dst_path = os.path.join(burst_dir, f"frame_{i}.png")
            shutil.copy2(src_path, dst_path)
            
        logger.debug("📦 Saved hanging burst for %s: %s (%d frames)", region, burst_dir, len(burst_frames))
    except Exception as exc:
        logger.warning("Failed to save hanging burst for %s: %s", region, exc)

# ── helper ───────────────────────────────────────────────────────────────────
def _copy_evidence(burst: List[Dict], tag: str) -> None:
    for f in burst:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        base = os.path.basename(f["path"])
        dst  = os.path.join(EVIDENCE_DIR, f"{ts}_{tag}_{base}")
        try:
            shutil.copy2(f["path"], dst)
            logger.debug("📥 Copied evidence %s → %s", f["path"], dst)
        except Exception as exc:
            logger.warning("Could not copy evidence %s : %s", f["path"], exc)

# ── core loop ────────────────────────────────────────────────────────────────
def update_states() -> None:
    global DELETE_QUEUE
    frames = download_latest_images(drive)      # region→{"path","id"} - no folder_id needed
    logger.debug("Pulled %d regions", len(frames))
    now = datetime.now(timezone.utc)

    # Store rack detections for cross-zone analysis
    rack_detections = {}

    for region, frame in frames.items():
        img_path = frame["path"]
        file_id  = frame["id"]

        # ── Save current status image ──────────────────────────────────────
        save_current_status_image(region, img_path)

        # ── person/door detection ------------------------------------------
        detect_result   = detect_person_in_booth(img_path)
        in_use          = detect_result.get("in_booth", False)
        door_closed     = detect_result.get("door_closed")
        person_detected = detect_result.get("person_detected")
        logger.debug("%s detect → %s", region, detect_result)

        # ── booth FSM ------------------------------------------------------
        prev_in_use = booth_state[region]["in_use"]
        update_booth_status(region, in_use, door_closed, person_detected)

        if in_use and not prev_in_use:
            booth_state[region].update({"in_use": True, "entered_at": now})
            add_booth_history(region, "OCCUPIED",
                              door_closed=door_closed,
                              person_detected=person_detected)
            logger.info("👤 entered %s", region)

        elif not in_use and prev_in_use:
            entered_at = booth_state[region]["entered_at"]
            if entered_at:
                elapsed = (now - entered_at).total_seconds()
                update_time_spent(region, elapsed)
                add_booth_history(region, "EMPTY",
                                  duration_seconds=elapsed,
                                  door_closed=door_closed,
                                  person_detected=person_detected)
                logger.info("👤 exited %s (%.0fs)", region, elapsed)
            booth_state[region].update({"in_use": False, "entered_at": None})

        # ── rack buffers (always fill) -------------------------------------
        RACK_BUFFERS[region].append(frame)
        if len(RACK_BUFFERS[region]) == FRAME_SEQUENCE:
            try:
                burst_paths = [f["path"] for f in RACK_BUFFERS[region]]
                # Use enhanced rack detection with global tracking
                positions = detect_racks(burst_paths, camera_id=region)
                if positions:
                    add_rack_positions(positions)
                    logger.debug("📈 Stored %d rack positions with global IDs [%s]", len(positions), region)
                    
                    # Store for cross-zone movement analysis
                    rack_detections[region] = positions
                    
                    # Log global rack assignments
                    for pos in positions:
                        global_id = pos.get("global_rack_id", "unassigned")
                        zone = pos.get("zone_description", "unknown")
                        moving = "🔄" if pos.get("moving", 0) else "📍"
                        logger.info(f"   └─ {moving} {pos['rack_id']} → {global_id} in {zone}")
                    
                    # ── Save rack burst if movement detected ──────────────────
                    movement_detected = any(p.get("moving", 0) for p in positions)
                    if movement_detected:
                        timestamp = positions[0]["timestamp"]
                        save_rack_burst(region, list(RACK_BUFFERS[region]), timestamp)
                        logger.info("🔄 Rack movement detected in %s, burst saved", region)
                        
            except Exception:
                logger.exception("Enhanced rack detection failed [%s]", region)
            finally:
                RACK_BUFFERS[region].clear()

        # ── Enhanced action buffers (only if in_use) -----------------------
        ACTION_BUFFERS[region].append(frame)
        if len(ACTION_BUFFERS[region]) == FRAME_SEQUENCE:  # ← REMOVED "and in_use"
            try:
                paths  = [f["path"] for f in ACTION_BUFFERS[region]]
                
                if USE_ENHANCED_DETECTION:
                    # Use enhanced detection with operation types and explanations
                    events = detect_rack_actions(paths, camera_id=region)
                else:
                    # Fallback to legacy detection for testing
                    from action_detection import detect_rack_actions_legacy
                    events = detect_rack_actions_legacy(paths, camera_id=region)
                
                if events:
                    add_rack_events(events)
                    logger.info("📝 %d enhanced rack-events stored [%s]", len(events), region)
                    
                    # Log enhanced detection details
                    if USE_ENHANCED_DETECTION:
                        for event in events:
                            operation_type = event.get("operation_type", "unknown")
                            forklift = "🚜" if event.get("forklift_detected") else ""
                            manual = "👤" if event.get("manual_detected") else ""
                            confidence = event.get("confidence", 0.0)
                            logger.info(f"   └─ {operation_type} {forklift}{manual}: {event.get('action', 'unknown')} (conf: {confidence:.2f})")
                
                # ── Enhanced hanging detection with operation types ────────────
                hanging_detected = False
                if events:
                    if USE_ENHANCED_DETECTION:
                        # Check for both manual and forklift hanging operations
                        hanging_operations = ["manual_hanging", "forklift_hanging"]
                        hanging_detected = any(
                            e.get("operation_type") in hanging_operations or
                            (e.get("action", "").lower().find("hanging") != -1)
                            for e in events
                        )
                        
                        if hanging_detected:
                            # Enhanced logging for hanging detection
                            hanging_types = [e.get("operation_type") for e in events 
                                           if e.get("operation_type") in hanging_operations]
                            if hanging_types:
                                logger.info("🔗 Hanging detected [%s]: %s", region, ', '.join(hanging_types))
                    else:
                        # Legacy hanging detection
                        hanging_detected = any(e.get("action", "").lower().find("hanging") != -1 for e in events)
                    
                    if hanging_detected:
                        timestamp = events[0]["timestamp"]
                        save_hanging_burst(region, list(ACTION_BUFFERS[region]), timestamp)
                
                keep_burst = hanging_detected
                for f in ACTION_BUFFERS[region]:
                    fid = f["id"]
                    if not fid:
                        continue
                    if keep_burst:
                        KEEP_FILE_IDS.add(fid)
                    else:
                        DELETE_QUEUE.append(fid)
                if keep_burst:
                    _copy_evidence(ACTION_BUFFERS[region], region)
            except Exception:
                logger.exception("Enhanced rack action detection failed [%s]", region)
            finally:
                ACTION_BUFFERS[region].clear()
        elif file_id and not in_use:
            DELETE_QUEUE.append(file_id)

    # ── cross-zone movement detection (once per cycle) ---------------------
    try:
        # Check if we have rack detections from both key zones
        general_labor_racks = rack_detections.get("general_labor", [])
        powder_booth_racks = rack_detections.get("powder_booth", [])
        
        if general_labor_racks or powder_booth_racks:
            from rack_detection import detect_cross_zone_movements
            movements = detect_cross_zone_movements(general_labor_racks, powder_booth_racks)
            if movements:
                logger.info("🔄 Detected %d cross-zone rack movements this cycle", len(movements))
                for movement in movements:
                    movement_type = movement.get('type', 'unknown')
                    rack_id = movement.get('global_rack_id', 'unknown')
                    description = movement.get('description', 'no description')
                    logger.info(f"   └─ {movement_type}: {rack_id} - {description}")
        
    except Exception:
        logger.exception("Cross-zone movement detection failed")

    # ── cleanup Drive (only if enabled) ------------------------------------
    if ENABLE_DELETION and DELETE_QUEUE:
        to_delete = [fid for fid in DELETE_QUEUE if fid not in KEEP_FILE_IDS]
        if to_delete:
            delete_files(drive, to_delete)
            logger.info("🗑️ Deleted %d files from Drive", len(to_delete))
        DELETE_QUEUE = []
    elif DELETE_QUEUE:
        logger.info("📦 Preserving %d files in Drive (deletion disabled)", len(DELETE_QUEUE))
        DELETE_QUEUE = []  # Clear the queue but don't delete

    write_json_files()


def write_json_files() -> None:
    booth_data = get_all_booth_status()
    booth_json, sandblast_json = {}, {}
    for b in booth_data:
        reg = b["region"]
        booth_json[reg] = {
            "in_use": b["in_use"],
            "time_spent_minutes": round((b.get("time_spent_today") or 0) / 60, 2),
        }
        if reg == "sandblast":
            sandblast_json = {
                "door_closed": b.get("door_closed", False),
                "person_detected": b.get("person_detected", False),
                "last_check": b.get("last_updated"),
            }
    with open("booth_state.json", "w") as f:
        json.dump(booth_json, f, indent=2)
    if sandblast_json:
        with open("sandblast_state.json", "w") as f:
            json.dump(sandblast_json, f, indent=2)
    logger.debug("Dashboard JSON written (%d booths)", len(booth_json))


def bg_loop() -> None:
    logger.info("Monitor tick every %ds (deletion %s, enhanced detection %s)", 
                CHECK_INTERVAL, 
                "ENABLED" if ENABLE_DELETION else "DISABLED",
                "ENABLED" if USE_ENHANCED_DETECTION else "DISABLED")
    
    # Validate enhanced detection setup
    if USE_ENHANCED_DETECTION:
        examples_available = validate_enhanced_detection()
        if not examples_available:
            logger.info("💡 To improve detection accuracy, add example images:")
            logger.info("   mkdir example_frames")
            logger.info("   # Add manual_hanging_1-4.png, forklift_hanging_1-4.png, etc.")
    
    # Validate rack detection setup
    rack_detection_available = validate_rack_detection()
    if not rack_detection_available:
        logger.info("💡 To improve rack tracking accuracy, add example images:")
        logger.info("   mkdir rack_examples")
        logger.info("   # Add general_labor_stable_1-4.png, powder_booth_stable_1-4.png, etc.")
    
    write_json_files()
    while True:
        try:
            update_states()
        except Exception:
            logger.exception("Monitor loop crash – continuing")
        time.sleep(CHECK_INTERVAL)
        
# ── Flask routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Redirect root URL to the dashboard."""
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    """Enhanced dashboard route with image review"""
    return render_template("dashboard.html")

@app.route("/status")
def status_api():
    try:
        return jsonify({b["region"]: b for b in get_all_booth_status()})
    except Exception as e:
        logger.error("Status API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/history")
def history():
    try:
        return jsonify(get_booth_history(limit=50))
    except Exception as e:
        logger.error("History API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/history/<region>")
def region_history(region):
    try:
        return jsonify(get_booth_history(region=region, limit=50))
    except Exception as e:
        logger.error("Region history error: %s", e)
        return jsonify({"error": str(e)}), 500

# ── Image serving routes ─────────────────────────────────────────────────────
@app.route("/api/current-image/<region>")
def serve_current_image(region):
    """Serve the latest current status image for a region"""
    try:
        image_path = os.path.join(CURRENT_IMAGES_DIR, f"{region}_current.png")
        if os.path.exists(image_path):
            response = send_file(image_path, mimetype='image/png')
            # Prevent caching to ensure fresh images
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        else:
            abort(404)
    except Exception as e:
        logger.error("Error serving current image for %s: %s", region, e)
        abort(500)

@app.route("/api/rack-burst/<camera>/<timestamp>/<int:frame_num>")
def serve_rack_burst_frame(camera, timestamp, frame_num):
    """Serve a specific frame from a rack movement burst"""
    try:
        if frame_num < 1 or frame_num > 4:
            abort(400)
            
        safe_timestamp = timestamp.replace(":", "-").replace("T", "_").replace("Z", "")
        image_path = os.path.join(RACK_BURSTS_DIR, camera, safe_timestamp, f"frame_{frame_num}.png")
        
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/png')
        else:
            abort(404)
    except Exception as e:
        logger.error("Error serving rack burst frame %d for %s/%s: %s", frame_num, camera, timestamp, e)
        abort(500)

@app.route("/api/hanging-burst/<camera>/<timestamp>/<int:frame_num>")
def serve_hanging_burst_frame(camera, timestamp, frame_num):
    """Serve a specific frame from a hanging parts burst"""
    try:
        if frame_num < 1 or frame_num > 4:
            abort(400)
            
        safe_timestamp = timestamp.replace(":", "-").replace("T", "_").replace("Z", "")
        image_path = os.path.join(HANGING_BURSTS_DIR, camera, safe_timestamp, f"frame_{frame_num}.png")
        
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/png')
        else:
            abort(404)
    except Exception as e:
        logger.error("Error serving hanging burst frame %d for %s/%s: %s", frame_num, camera, timestamp, e)
        abort(500)

# ── API routes for dashboard ────────────────────────────────────────────────
@app.route("/api/booth-status")
def booth_status_api():
    """Get all booth status data"""
    try:
        booth_data = get_all_booth_status()
        return jsonify(booth_data)
    except Exception as e:
        logger.error("Booth status API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/rack-positions")
def rack_positions_api():
    """Get latest rack position data"""
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM rack_positions 
            ORDER BY timestamp DESC 
            LIMIT 100
        """)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify(rows)
    except Exception as e:
        logger.error("Rack positions API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/rack-events")  
def rack_events_api():
    """Get latest rack events data with enhanced fields"""
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM rack_events 
            ORDER BY timestamp DESC 
            LIMIT 100
        """)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify(rows)
    except Exception as e:
        logger.error("Rack events API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/rack-positions/<camera_id>")
def rack_positions_by_camera(camera_id):
    """Get rack positions for specific camera"""
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM rack_positions 
            WHERE camera_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 50
        """, (camera_id,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify(rows)
    except Exception as e:
        logger.error("Camera rack positions API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/rack-events/<camera_id>")
def rack_events_by_camera(camera_id):
    """Get rack events for specific camera with enhanced data"""
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM rack_events 
            WHERE camera_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 50
        """, (camera_id,))
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return jsonify(rows)
    except Exception as e:
        logger.error("Camera rack events API error: %s", e)
        return jsonify({"error": str(e)}), 500

# ── Enhanced API routes ──────────────────────────────────────────────────────
@app.route("/api/rack-events/by-type/<operation_type>")
def rack_events_by_type(operation_type):
    """Get rack events filtered by operation type"""
    try:
        events = get_rack_events_by_type(operation_type=operation_type, limit=50)
        return jsonify(events)
    except Exception as e:
        logger.error("Rack events by type API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/operation-statistics")
def operation_statistics():
    """Get statistics on manual vs forklift operations"""
    try:
        hours = int(request.args.get('hours', 24))
        stats = get_operation_statistics(hours=hours)
        return jsonify(stats)
    except Exception as e:
        logger.error("Operation statistics API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/events-with-explanations")
def events_with_explanations():
    """Get recent events that have LLM explanations"""
    try:
        camera_id = request.args.get('camera_id')
        limit = int(request.args.get('limit', 20))
        events = get_events_with_explanations(camera_id=camera_id, limit=limit)
        return jsonify(events)
    except Exception as e:
        logger.error("Events with explanations API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats")
def stats_api():
    """Get enhanced summary statistics"""
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # Get rack event counts by action
        cur.execute("""
            SELECT action, COUNT(*) as count 
            FROM rack_events 
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY action
        """)
        action_counts = {row[0]: row[1] for row in cur.fetchall()}
        
        # Get operation type counts (enhanced)
        cur.execute("""
            SELECT operation_type, COUNT(*) as count 
            FROM rack_events 
            WHERE timestamp > datetime('now', '-24 hours')
            AND operation_type IS NOT NULL
            GROUP BY operation_type
        """)
        operation_type_counts = {row[0]: row[1] for row in cur.fetchall()}
        
        # Get rack position counts by camera
        cur.execute("""
            SELECT camera_id, COUNT(*) as count 
            FROM rack_positions 
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY camera_id
        """)
        position_counts = {row[0]: row[1] for row in cur.fetchall()}
        
        # Get moving rack counts
        cur.execute("""
            SELECT camera_id, 
                   SUM(CASE WHEN moving = 1 THEN 1 ELSE 0 END) as moving_count,
                   COUNT(*) as total_count
            FROM rack_positions 
            WHERE timestamp > datetime('now', '-1 hours')
            GROUP BY camera_id
        """)
        movement_stats = {}
        for row in cur.fetchall():
            movement_stats[row[0]] = {
                "moving": row[1],
                "total": row[2],
                "moving_percentage": round((row[1] / row[2]) * 100, 1) if row[2] > 0 else 0
            }
        
        # Enhanced: Manual vs Forklift statistics
        cur.execute("""
            SELECT 
                SUM(manual_detected) as manual_operations,
                SUM(forklift_detected) as forklift_operations,
                COUNT(*) as total_operations
            FROM rack_events 
            WHERE timestamp > datetime('now', '-24 hours')
        """)
        detection_stats = dict(cur.fetchone()) if cur.rowcount > 0 else {}
        
        conn.close()
        
        return jsonify({
            "action_counts": action_counts,
            "operation_type_counts": operation_type_counts,  # Enhanced
            "position_counts": position_counts,
            "movement_stats": movement_stats,
            "detection_stats": detection_stats,  # Enhanced
            "enhanced_detection_enabled": USE_ENHANCED_DETECTION
        })
    except Exception as e:
        logger.error("Enhanced stats API error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Enhanced Rack Detection API Routes ───────────────────────────────────
@app.route("/api/rack-detection-status")
def rack_detection_status():
    """Get status of enhanced rack detection system"""
    try:
        validation = validate_rack_detection_setup()
        return jsonify(validation)
    except Exception as e:
        logger.error("Rack detection status error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/rack-summary")
def rack_summary():
    """Get comprehensive rack tracking summary"""
    try:
        summary = get_rack_detection_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error("Rack summary error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/global-racks")
def global_racks():
    """Get current global rack registry"""
    try:
        from database import get_global_rack_state
        racks = get_global_rack_state()
        return jsonify(racks)
    except Exception as e:
        logger.error("Global racks API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/rack-movements")
def rack_movements():
    """Get recent rack movements between zones"""
    try:
        from database import get_recent_movements
        hours = int(request.args.get('hours', 24))
        movements = get_recent_movements(hours)
        return jsonify(movements)
    except Exception as e:
        logger.error("Rack movements API error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/admin/cleanup-rack-data", methods=["POST"])
def cleanup_rack_data():
    """Admin endpoint to clean up old rack tracking data"""
    try:
        from rack_detection import cleanup_old_rack_data
        hours = int(request.json.get('hours', 24)) if request.is_json else 24
        
        cleanup_results = cleanup_old_rack_data(hours)
        
        return jsonify({
            "status": "success",
            "results": cleanup_results
        })
        
    except Exception as e:
        logger.error(f"Error during rack data cleanup: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
# ── Admin routes for deletion control ────────────────────────────────────────
@app.route("/admin/deletion-status")
def deletion_status():
    """Check if deletion is enabled and show queue status"""
    try:
        return jsonify({
            "deletion_enabled": ENABLE_DELETION,
            "enhanced_detection_enabled": USE_ENHANCED_DETECTION,
            "files_in_queue": len(DELETE_QUEUE),
            "evidence_files": len(KEEP_FILE_IDS),
            "message": f"Deletion is {'ENABLED' if ENABLE_DELETION else 'DISABLED'}, Enhanced detection is {'ENABLED' if USE_ENHANCED_DETECTION else 'DISABLED'}"
        })
    except Exception as e:
        logger.error("Deletion status error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/admin/manual-cleanup", methods=["POST"])
def manual_cleanup():
    """Manual Drive cleanup - only call when you're ready to delete old files"""
    try:
        if not drive:
            return jsonify({"error": "Google Drive not available"}), 500
            
        # Get the current delete queue
        to_delete = [fid for fid in DELETE_QUEUE if fid not in KEEP_FILE_IDS]
        
        if to_delete:
            delete_files(drive, to_delete)
            DELETE_QUEUE.clear()
            return jsonify({
                "status": "success", 
                "deleted": len(to_delete),
                "message": f"Manually deleted {len(to_delete)} files from Drive"
            })
        else:
            return jsonify({
                "status": "success", 
                "deleted": 0,
                "message": "No files in queue to delete"
            })
            
    except Exception as e:
        logger.error("Manual cleanup error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/admin/enhanced-detection-status")
def enhanced_detection_status():
    """Get status of enhanced detection setup"""
    try:
        validation = validate_examples_setup()
        
        # Handle different return formats from validate_examples_setup
        example_status = validation.get("example_status", "Unknown")
        if isinstance(example_status, dict):
            # If it's a dict, convert to string summary
            example_status = "Configuration details available"
        
        return jsonify({
            "enhanced_detection_enabled": USE_ENHANCED_DETECTION,
            "examples_configured": validation.get("examples_configured", False),
            "available_examples": validation.get("available_examples", []),
            "example_status": example_status,
            "recommendation": validation.get("recommendation", "No recommendation")
        })
    except Exception as e:
        logger.error("Enhanced detection status error: %s", e)
        # Return a safe response instead of 500 error
        return jsonify({
            "enhanced_detection_enabled": USE_ENHANCED_DETECTION,
            "examples_configured": False,
            "available_examples": [],
            "example_status": "Error",
            "recommendation": f"Error checking status: {str(e)}"
        })
        
# Add these routes to your existing app.py file

@app.route("/logs")
def log_viewer():
    """Serve the log viewer HTML interface"""
    return render_template("log_viewer.html")

@app.route("/api/llm-logs")
def get_llm_logs():
    """API endpoint to get parsed LLM interaction logs"""
    try:
        hours = int(request.args.get('hours', 24))  # Default last 24 hours
        module_filter = request.args.get('module', '')
        
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return jsonify([])
        
        # Find log files from the last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        log_files = []
        
        for log_file in logs_dir.glob("llm_interactions_*.log"):
            try:
                # Extract date from filename: llm_interactions_20241201.log
                date_str = log_file.stem.split('_')[-1]
                file_date = datetime.strptime(date_str, "%Y%m%d")
                if file_date >= cutoff_time.replace(hour=0, minute=0, second=0):
                    log_files.append(log_file)
            except (ValueError, IndexError):
                # Skip files that don't match the expected format
                continue
        
        # Parse log entries
        parsed_logs = []
        for log_file in sorted(log_files):
            try:
                parsed_logs.extend(parse_log_file(log_file, module_filter))
            except Exception as e:
                logger.error(f"Failed to parse log file {log_file}: {e}")
        
        # Sort by timestamp (newest first)
        parsed_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Limit to last 1000 entries to prevent overwhelming the browser
        if len(parsed_logs) > 1000:
            parsed_logs = parsed_logs[:1000]
        
        return jsonify(parsed_logs)
        
    except Exception as e:
        logger.error(f"Error serving LLM logs: {e}")
        return jsonify({"error": str(e)}), 500

def parse_log_file(log_file: Path, module_filter: str = '') -> list:
    """
    Improved log file parser that extracts all the detailed information
    """
    
    parsed_entries = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by the main section markers
        sections = re.split(r'=== (\w+) (REQUEST|RESPONSE|PARSING|ERROR) ===', content)
        
        i = 1
        while i < len(sections):
            if i + 2 >= len(sections):
                break
                
            module_name = sections[i].lower()
            section_type = sections[i + 1]
            section_content = sections[i + 2]
            
            # Apply module filter
            if module_filter and module_name != module_filter:
                i += 3
                continue
            
            # Extract timestamp from the section
            timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', section_content)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                # Convert to ISO format for JavaScript
                try:
                    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    iso_timestamp = dt.isoformat()
                except ValueError:
                    iso_timestamp = timestamp
            else:
                iso_timestamp = datetime.now().isoformat()
            
            entry = {
                'timestamp': iso_timestamp,
                'module': module_name,
                'type': section_type,
                'raw_content': section_content
            }
            
            # Extract fields based on section type
            if section_type == 'REQUEST':
                entry.update(extract_request_details(section_content))
            elif section_type == 'RESPONSE':
                entry.update(extract_response_details(section_content))
            elif section_type == 'PARSING':
                entry.update(extract_parsing_details(section_content))
            elif section_type == 'ERROR':
                entry.update(extract_error_details(section_content))
            
            parsed_entries.append(entry)
            i += 3
            
    except Exception as e:
        logger.error(f"Error parsing log file {log_file}: {e}")
    
    return parsed_entries

def extract_request_details(content: str) -> dict:
    """Extract detailed request information"""
    details = {}
    
    # Extract basic fields
    model_match = re.search(r'Model: (.+)', content)
    if model_match:
        details['model'] = model_match.group(1)
    
    temp_match = re.search(r'Temperature: (.+)', content)
    if temp_match:
        details['temperature'] = temp_match.group(1)
    
    tokens_match = re.search(r'Max tokens: (.+)', content)
    if tokens_match:
        details['max_tokens'] = tokens_match.group(1)
    
    images_match = re.search(r'Images attached: (\d+)', content)
    if images_match:
        details['image_count'] = int(images_match.group(1))
    
    # Extract prompt (look for the prompt preview or full payload)
    prompt_match = re.search(r'Prompt preview: (.+?)(?=\n\d{4}-\d{2}-\d{2}|\nFull payload|\n===|$)', content, re.DOTALL)
    if prompt_match:
        details['prompt'] = prompt_match.group(1).strip()
    
    # Try to extract JSON payload
    payload_match = re.search(r'Full payload: (\{.+?\})\n', content, re.DOTALL)
    if payload_match:
        try:
            payload_str = payload_match.group(1)
            # Handle the case where it might be truncated or malformed
            details['payload'] = json.loads(payload_str)
        except json.JSONDecodeError:
            details['payload'] = {"error": "Failed to parse payload JSON"}
    
    return details

def extract_response_details(content: str) -> dict:
    """Extract detailed response information"""
    details = {}
    
    # Extract processing time
    time_match = re.search(r'Processing time: ([\d.]+)s', content)
    if time_match:
        details['processing_time'] = float(time_match.group(1))
    
    # Extract token usage
    usage_match = re.search(r"Token usage: (\{[^}]+\})", content)
    if usage_match:
        try:
            usage_str = usage_match.group(1)
            details['usage'] = json.loads(usage_str.replace("'", '"'))
        except json.JSONDecodeError:
            details['usage'] = {}
    
    # Extract raw response text
    raw_match = re.search(r'Raw response text: (.+?)(?=\nFull response|\n\d{4}-\d{2}-\d{2}|\n===|$)', content, re.DOTALL)
    if raw_match:
        details['raw_text'] = raw_match.group(1).strip()
    
    # Extract full response JSON
    response_match = re.search(r'Full response: (\{.+?\})\n', content, re.DOTALL)
    if response_match:
        try:
            response_str = response_match.group(1)
            details['response_data'] = json.loads(response_str)
        except json.JSONDecodeError:
            details['response_data'] = {"error": "Failed to parse response JSON"}
    
    return details

def extract_parsing_details(content: str) -> dict:
    """Extract parsing attempt information"""
    details = {}
    
    # Extract parsing method
    method_match = re.search(r'Parsing method: (.+)', content)
    if method_match:
        details['parsing_method'] = method_match.group(1)
    
    # Extract success status
    success_match = re.search(r'Parsing successful: (True|False)', content)
    if success_match:
        details['success'] = success_match.group(1) == 'True'
    
    # Extract raw text being parsed
    raw_match = re.search(r'Raw text to parse: (.+?)(?=\nParsed result|\n\d{4}-\d{2}-\d{2}|\n===|$)', content, re.DOTALL)
    if raw_match:
        details['raw_text'] = raw_match.group(1).strip()
    
    # Extract parsed result
    result_match = re.search(r'Parsed result: (.+?)(?=\n\d{4}-\d{2}-\d{2}|\n===|$)', content, re.DOTALL)
    if result_match:
        result_str = result_match.group(1).strip()
        try:
            if result_str.startswith('[') or result_str.startswith('{'):
                details['parsed_result'] = json.loads(result_str)
            else:
                details['parsed_result'] = result_str
        except json.JSONDecodeError:
            details['parsed_result'] = result_str
    
    return details

def extract_error_details(content: str) -> dict:
    """Extract error information"""
    details = {}
    
    # Extract error type
    type_match = re.search(r'Error type: (.+)', content)
    if type_match:
        details['error_type'] = type_match.group(1)
    
    # Extract error message
    msg_match = re.search(r'Error message: (.+)', content)
    if msg_match:
        details['error_message'] = msg_match.group(1)
    
    # Extract context
    context_match = re.search(r'Error context: (\{.+?\})', content, re.DOTALL)
    if context_match:
        try:
            context_str = context_match.group(1)
            details['context'] = json.loads(context_str)
        except json.JSONDecodeError:
            details['context'] = {}
    
    return details

@app.route("/api/llm-logs/latest")
def get_latest_llm_logs():
    """Get just the most recent LLM interactions (last 10)"""
    try:
        logs_dir = Path("logs")
        if not logs_dir.exists():
            return jsonify([])
        
        # Get today's log file
        today = datetime.now().strftime("%Y%m%d")
        today_log = logs_dir / f"llm_interactions_{today}.log"
        
        if not today_log.exists():
            return jsonify([])
        
        # Parse and get last 10 entries
        parsed_logs = parse_log_file(today_log)
        latest_logs = sorted(parsed_logs, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
        
        return jsonify(latest_logs)
        
    except Exception as e:
        logger.error(f"Error getting latest LLM logs: {e}")
        return jsonify({"error": str(e)}), 500
    
# Add these routes to your app.py for serving saved LLM images

@app.route("/api/llm-images/<path:filename>")
def serve_llm_image(filename):
    """Serve saved LLM images"""
    try:
        images_dir = Path("logs/images")
        image_path = images_dir / filename
        
        # Security check - ensure the file is within the images directory
        if not str(image_path.resolve()).startswith(str(images_dir.resolve())):
            abort(403)  # Forbidden
        
        if image_path.exists() and image_path.is_file():
            return send_file(image_path)
        else:
            abort(404)
            
    except Exception as e:
        logger.error(f"Error serving LLM image {filename}: {e}")
        abort(500)

@app.route("/api/llm-images")
def list_llm_images():
    """List all saved LLM images with metadata"""
    try:
        images_dir = Path("logs/images")
        if not images_dir.exists():
            return jsonify([])
        
        images = []
        for image_file in images_dir.glob("*.png"):
            try:
                stat = image_file.stat()
                
                # Parse filename to extract metadata
                # Format: module_camera_timestamp_frame_N.ext
                name_parts = image_file.stem.split('_')
                
                metadata = {
                    "filename": image_file.name,
                    "path": f"/api/llm-images/{image_file.name}",
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
                
                # Try to extract metadata from filename
                if len(name_parts) >= 4:
                    metadata.update({
                        "module": name_parts[0],
                        "camera": name_parts[1] if len(name_parts) > 4 else "",
                        "timestamp": name_parts[-3] + "_" + name_parts[-2] if len(name_parts) >= 4 else "",
                        "frame": name_parts[-1] if name_parts[-1].startswith('frame') else "unknown"
                    })
                
                images.append(metadata)
                
            except Exception as e:
                logger.warning(f"Error processing image file {image_file}: {e}")
                continue
        
        # Also check JPG files
        for image_file in images_dir.glob("*.jpg"):
            try:
                stat = image_file.stat()
                name_parts = image_file.stem.split('_')
                
                metadata = {
                    "filename": image_file.name,
                    "path": f"/api/llm-images/{image_file.name}",
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
                
                if len(name_parts) >= 4:
                    metadata.update({
                        "module": name_parts[0],
                        "camera": name_parts[1] if len(name_parts) > 4 else "",
                        "timestamp": name_parts[-3] + "_" + name_parts[-2] if len(name_parts) >= 4 else "",
                        "frame": name_parts[-1] if name_parts[-1].startswith('frame') else "unknown"
                    })
                
                images.append(metadata)
                
            except Exception as e:
                logger.warning(f"Error processing image file {image_file}: {e}")
                continue
        
        # Sort by creation time (newest first)
        images.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify(images)
        
    except Exception as e:
        logger.error(f"Error listing LLM images: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/llm-images/by-request/<module>/<timestamp>")
def get_images_for_request(module, timestamp):
    """Get all images for a specific LLM request"""
    try:
        from llm_tracking import get_saved_images_for_timestamp
        
        # Extract camera from request args if provided
        camera_id = request.args.get('camera', '')
        
        image_paths = get_saved_images_for_timestamp(module, timestamp, camera_id)
        
        images = []
        for img_path in image_paths:
            img_file = Path(img_path)
            if img_file.exists():
                stat = img_file.stat()
                images.append({
                    "filename": img_file.name,
                    "path": f"/api/llm-images/{img_file.name}",
                    "size": stat.st_size,
                    "full_path": str(img_path)
                })
        
        return jsonify(images)
        
    except Exception as e:
        logger.error(f"Error getting images for request {module}/{timestamp}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/images")
def image_gallery():
    """Serve a simple image gallery page"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Images Gallery</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .controls {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .controls button {
            background: #007cba;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .controls button:hover {
            background: #005a87;
        }
        
        .controls select, .controls input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .image-card {
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        .image-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            cursor: pointer;
        }
        
        .image-info {
            padding: 15px;
        }
        
        .image-info h3 {
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #333;
        }
        
        .image-info p {
            margin: 5px 0;
            font-size: 12px;
            color: #666;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }
        
        .modal-content {
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            margin-top: 5%;
        }
        
        .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }
        
        .no-images {
            text-align: center;
            padding: 40px;
            color: #6c757d;
            background: white;
            border-radius: 8px;
        }
        
        .frame-sequence {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .frame-sequence img {
            width: 60px;
            height: 45px;
            border: 2px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .frame-sequence img:hover {
            border-color: #007cba;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 LLM Images Gallery</h1>
        
        <div class="controls">
            <button onclick="loadImages()">🔄 Refresh</button>
            <button onclick="clearGallery()">🗑️ Clear</button>
            
            <select id="moduleFilter">
                <option value="">All Modules</option>
                <option value="action_detection">Action Detection</option>
                <option value="rack_detection">Rack Detection</option>
                <option value="lm_detection">LM Detection</option>
            </select>
            
            <select id="cameraFilter">
                <option value="">All Cameras</option>
                <option value="general_labor">General Labor</option>
                <option value="sandblast">Sandblast</option>
                <option value="powder_booth">Powder Booth</option>
            </select>
            
            <input type="text" id="searchInput" placeholder="Search by filename..." onkeyup="filterImages()">
            
            <span id="imageCount">0 images</span>
        </div>
        
        <div id="imageContainer">
            <div class="no-images">
                Click "Refresh" to load saved LLM images
            </div>
        </div>
    </div>
    
    <!-- Modal for full-size image viewing -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        let allImages = [];

        async function loadImages() {
            try {
                const response = await fetch('/api/llm-images');
                const images = await response.json();
                
                allImages = images;
                displayImages(images);
                updateImageCount(images.length);
                
                console.log(`Loaded ${images.length} images`);
            } catch (error) {
                console.error('Failed to load images:', error);
                document.getElementById('imageContainer').innerHTML = 
                    '<div class="no-images">Failed to load images. Make sure the server is running.</div>';
            }
        }

        function displayImages(images) {
            const container = document.getElementById('imageContainer');
            
            if (images.length === 0) {
                container.innerHTML = '<div class="no-images">No images found</div>';
                return;
            }

            // Group images by request (same timestamp and module)
            const grouped = groupImagesByRequest(images);
            
            container.innerHTML = `<div class="image-grid">${
                Object.values(grouped).map(group => generateImageCard(group)).join('')
            }</div>`;
        }

        function groupImagesByRequest(images) {
            const groups = {};
            
            images.forEach(img => {
                const key = `${img.module}_${img.camera}_${img.timestamp}`;
                if (!groups[key]) {
                    groups[key] = [];
                }
                groups[key].push(img);
            });
            
            return groups;
        }

        function generateImageCard(imageGroup) {
            const mainImage = imageGroup[0];
            const timestamp = new Date(mainImage.created).toLocaleString();
            
            return `
                <div class="image-card">
                    <img src="${mainImage.path}" onclick="openModal('${mainImage.path}')" alt="${mainImage.filename}">
                    <div class="image-info">
                        <h3>${mainImage.module.toUpperCase()} - ${mainImage.camera}</h3>
                        <p><strong>Time:</strong> ${timestamp}</p>
                        <p><strong>Frames:</strong> ${imageGroup.length}</p>
                        <p><strong>Size:</strong> ${formatFileSize(mainImage.size)}</p>
                        <div class="frame-sequence">
                            ${imageGroup.map(img => 
                                `<img src="${img.path}" onclick="openModal('${img.path}')" title="${img.filename}">`
                            ).join('')}
                        </div>
                    </div>
                </div>
            `;
        }

        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        function openModal(imagePath) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = imagePath;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        function filterImages() {
            const moduleFilter = document.getElementById('moduleFilter').value;
            const cameraFilter = document.getElementById('cameraFilter').value;
            const searchText = document.getElementById('searchInput').value.toLowerCase();
            
            let filtered = allImages.filter(img => {
                const matchesModule = !moduleFilter || img.module === moduleFilter;
                const matchesCamera = !cameraFilter || img.camera === cameraFilter;
                const matchesSearch = !searchText || img.filename.toLowerCase().includes(searchText);
                
                return matchesModule && matchesCamera && matchesSearch;
            });
            
            displayImages(filtered);
            updateImageCount(filtered.length);
        }

        function clearGallery() {
            document.getElementById('imageContainer').innerHTML = 
                '<div class="no-images">Gallery cleared. Click "Refresh" to reload.</div>';
            updateImageCount(0);
        }

        function updateImageCount(count) {
            document.getElementById('imageCount').textContent = `${count} images`;
        }

        // Set up event listeners
        document.getElementById('moduleFilter').addEventListener('change', filterImages);
        document.getElementById('cameraFilter').addEventListener('change', filterImages);

        // Close modal when clicking outside the image
        window.onclick = function(event) {
            const modal = document.getElementById('imageModal');
            if (event.target === modal) {
                closeModal();
            }
        }

        // Auto-refresh every 60 seconds
        setInterval(loadImages, 60000);
        
        // Load images on page load
        loadImages();
    </script>
</body>
</html>"""
    
    return html_content

@app.route("/admin/cleanup-images", methods=["POST"])
def cleanup_old_images():
    """Admin endpoint to clean up old saved images"""
    try:
        days_to_keep = int(request.json.get('days', 7)) if request.is_json else 7
        
        from llm_tracking import cleanup_old_images
        cleanup_old_images(days_to_keep)
         
        return jsonify({
            "status": "success",
            "message": f"Cleaned up images older than {days_to_keep} days"
        })
        
    except Exception as e:
        logger.error(f"Error during image cleanup: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/debug")
def debug_tool():
    # Return the HTML content directly or serve the file
    return render_template("rack_debug.html")  # if you put it in templates/
    # OR just return the HTML string directly

@app.route("/test")
def test():
    return "Flask is running & DB initialised with enhanced detection support."

# ── entrypoint ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Thread(target=bg_loop, daemon=True).start()
    logger.info("Flask at http://127.0.0.1:5002 (check /dashboard)")
    app.run(debug=False, port=5002)