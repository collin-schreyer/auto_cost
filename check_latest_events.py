#!/usr/bin/env python3
"""
Quick check to see the very latest events stored in the database
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

def check_latest():
    """Check the most recent events"""
    db_path = Path("booth_monitor.db")
    if not db_path.exists():
        print("❌ Database file not found")
        return
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Get total count
    cur.execute("SELECT COUNT(*) FROM rack_events")
    total = cur.fetchone()[0]
    print(f"📊 Total events in database: {total}")
    
    # Get events from the last hour
    cur.execute("""
        SELECT * FROM rack_events 
        WHERE timestamp > datetime('now', '-1 hours')
        ORDER BY timestamp DESC
    """)
    
    recent_events = [dict(row) for row in cur.fetchall()]
    
    if recent_events:
        print(f"🕐 Found {len(recent_events)} events in the last hour:")
        for event in recent_events:
            timestamp = event['timestamp']
            camera = event['camera_id']
            operation = event['operation_type']
            explanation = event['explanation'][:60] + "..." if len(event['explanation']) > 60 else event['explanation']
            print(f"   📝 {timestamp} | {camera} | {operation} | {explanation}")
    else:
        print("❌ No events found in the last hour")
    
    # Get the very latest events (last 5)
    print(f"\n🔍 Last 5 events stored:")
    cur.execute("""
        SELECT * FROM rack_events 
        ORDER BY id DESC
        LIMIT 5
    """)
    
    latest_events = [dict(row) for row in cur.fetchall()]
    
    for i, event in enumerate(latest_events, 1):
        timestamp = event['timestamp']
        camera = event['camera_id']
        operation = event['operation_type']
        action = event['action']
        explanation = event['explanation'][:50] + "..." if len(event['explanation']) > 50 else event['explanation']
        
        print(f"   {i}. {timestamp} | {camera} | {operation} | {action}")
        print(f"      {explanation}")
    
    conn.close()

if __name__ == "__main__":
    print("🔍 CHECKING LATEST EVENTS")
    print("=" * 50)
    check_latest()
    print("=" * 50)
    print("💡 If you see recent events, the fix is working!")
    print("💡 If no recent events, check if the app is running and processing images")