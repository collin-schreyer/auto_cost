#!/usr/bin/env python3
"""
Debug script to test the add_rack_events function and find why events aren't being stored
"""

import sys
import os
from datetime import datetime

# Add the current directory to the path so we can import from the project
sys.path.insert(0, os.getcwd())

def test_add_rack_events():
    """Test if the add_rack_events function works"""
    print("🧪 Testing add_rack_events function...")
    
    try:
        from database import add_rack_events
        print("✅ Successfully imported add_rack_events")
        
        # Create a test event similar to what's being detected
        test_event = {
            "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "camera_id": "test_camera",
            "rack_id": "test_rack",
            "global_rack_id": None,
            "people_count": 1,
            "action": "test hanging",
            "confidence": 0.85,
            "zone": "test_zone",
            "operation_type": "manual_hanging",
            "explanation": "TEST: Manual hanging detection test",
            "forklift_detected": 0,
            "manual_detected": 1,
        }
        
        print(f"📝 Test event: {test_event}")
        
        # Try to store the test event
        print("💾 Attempting to store test event...")
        add_rack_events([test_event])
        print("✅ Test event stored successfully!")
        
        # Verify it was stored
        from database import _connect
        conn = _connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM rack_events 
            WHERE explanation = 'TEST: Manual hanging detection test'
            ORDER BY timestamp DESC LIMIT 1
        """)
        
        stored_event = cur.fetchone()
        if stored_event:
            print("✅ Test event found in database!")
            print(f"   Stored at: {dict(stored_event)['timestamp']}")
        else:
            print("❌ Test event not found in database!")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error testing add_rack_events: {e}")
        import traceback
        traceback.print_exc()

def check_app_main_loop():
    """Check if the app's main loop is calling add_rack_events correctly"""
    print("\n🔍 Checking app.py main loop logic...")
    
    try:
        # Read the app.py file and look for add_rack_events calls
        with open("app.py", "r") as f:
            app_content = f.read()
        
        # Look for the add_rack_events call
        if "add_rack_events(events)" in app_content:
            print("✅ Found add_rack_events(events) call in app.py")
            
            # Look for the context around the call
            lines = app_content.split('\n')
            for i, line in enumerate(lines):
                if "add_rack_events(events)" in line:
                    print(f"📍 Found at line {i+1}:")
                    # Show context (5 lines before and after)
                    start = max(0, i-5)
                    end = min(len(lines), i+6)
                    for j in range(start, end):
                        marker = " >>> " if j == i else "     "
                        print(f"{marker}{j+1:4d}: {lines[j]}")
                    break
        else:
            print("❌ Could not find add_rack_events(events) call in app.py")
            print("🔍 Looking for similar patterns...")
            
            # Look for variations
            if "add_rack_events" in app_content:
                print("✅ Found add_rack_events references")
                lines = app_content.split('\n')
                for i, line in enumerate(lines):
                    if "add_rack_events" in line and "def " not in line:
                        print(f"     Line {i+1}: {line.strip()}")
            else:
                print("❌ No add_rack_events references found at all!")
        
    except Exception as e:
        print(f"❌ Error checking app.py: {e}")

def check_recent_logs():
    """Check for recent error logs that might explain the issue"""
    print("\n📝 Checking for recent error patterns...")
    
    try:
        # Look for log files
        log_files = []
        if os.path.exists("logs"):
            for file in os.listdir("logs"):
                if file.endswith(".log"):
                    log_files.append(os.path.join("logs", file))
        
        if not log_files:
            print("❌ No log files found in logs/ directory")
            return
        
        print(f"📁 Found {len(log_files)} log files")
        
        # Check the most recent log file for errors
        latest_log = max(log_files, key=os.path.getmtime)
        print(f"📄 Checking latest log: {latest_log}")
        
        with open(latest_log, "r") as f:
            lines = f.readlines()
        
        # Look for recent errors or storage-related messages
        recent_errors = []
        storage_messages = []
        
        for line in lines[-100:]:  # Check last 100 lines
            if "ERROR" in line or "Exception" in line:
                recent_errors.append(line.strip())
            if "add_rack_events" in line or "Enhanced rack action detection" in line:
                storage_messages.append(line.strip())
        
        if recent_errors:
            print(f"⚠️  Found {len(recent_errors)} recent errors:")
            for error in recent_errors[-5:]:  # Show last 5 errors
                print(f"     {error}")
        
        if storage_messages:
            print(f"📝 Found {len(storage_messages)} storage-related messages:")
            for msg in storage_messages[-5:]:  # Show last 5 messages
                print(f"     {msg}")
        
        if not recent_errors and not storage_messages:
            print("✅ No recent errors or storage issues found in logs")
    
    except Exception as e:
        print(f"❌ Error checking logs: {e}")

def check_permissions():
    """Check if there are permission issues with the database"""
    print("\n🔐 Checking database permissions...")
    
    try:
        db_path = "booth_monitor.db"
        if os.path.exists(db_path):
            stat = os.stat(db_path)
            print(f"📄 Database file: {db_path}")
            print(f"   Size: {stat.st_size} bytes")
            print(f"   Readable: {os.access(db_path, os.R_OK)}")
            print(f"   Writable: {os.access(db_path, os.W_OK)}")
            
            # Check if we can write to it
            from database import _connect
            conn = _connect()
            cur = conn.cursor()
            
            # Try a simple write operation
            cur.execute("SELECT COUNT(*) FROM rack_events")
            count_before = cur.fetchone()[0]
            print(f"   Current event count: {count_before}")
            
            conn.close()
            print("✅ Database is accessible and readable")
        else:
            print("❌ Database file does not exist!")
    
    except Exception as e:
        print(f"❌ Database permission error: {e}")

def main():
    print("🔍 DEBUGGING RACK EVENTS STORAGE ISSUE")
    print("=" * 60)
    print("We know events are being detected and parsed correctly,")
    print("but they're not making it to the database.")
    print("=" * 60)
    
    test_add_rack_events()
    check_app_main_loop()
    check_recent_logs()
    check_permissions()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS SUMMARY:")
    print("1. Events are being detected (✅ confirmed from logs)")
    print("2. JSON parsing works (✅ confirmed from logs)")
    print("3. Database schema is correct (✅ confirmed)")
    print("4. Previous events are stored (✅ confirmed)")
    print("5. NEW ISSUE: Recent events not reaching database")
    print("\n🔍 Most likely causes:")
    print("   - App crash/restart losing events in memory")
    print("   - Exception in add_rack_events() call")
    print("   - Database transaction not being committed")
    print("   - Logic condition preventing storage")
    
    print("\n💡 Next steps:")
    print("   1. Check if the app is currently running")
    print("   2. Look for exceptions in the app logs")
    print("   3. Add debug logging around add_rack_events() call")
    print("   4. Check if booth is 'in_use' (events only stored when booth occupied)")

if __name__ == "__main__":
    main()