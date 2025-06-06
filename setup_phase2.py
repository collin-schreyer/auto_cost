#!/usr/bin/env python3
"""
setup_phase2.py - Setup script for Phase 2 enhanced detection
------------------------------------------------------------
Creates directory structure and provides guidance for Phase 2 deployment.
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create required directories for Phase 2"""
    print("🏗️  Creating Phase 2 directory structure...")
    
    directories = [
        "example_frames",
        "review_images",
        "review_images/current_status", 
        "review_images/rack_bursts",
        "review_images/hanging_bursts",
        "evidence_hanging"
    ]
    
    created = []
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(directory)
            print(f"   ✅ Created: {directory}/")
        else:
            print(f"   📁 Exists: {directory}/")
    
    if created:
        print(f"\n✨ Created {len(created)} new directories")
    else:
        print("\n📋 All directories already exist")

def check_example_images():
    """Check which example images need to be added"""
    print("\n📸 Checking example images...")
    
    example_types = [
        "manual_hanging",
        "forklift_hanging", 
        "manual_removing",
        "forklift_removing",
        "other_activity"
    ]
    
    missing_examples = []
    for example_type in example_types:
        missing_frames = []
        for i in range(1, 5):
            image_path = Path(f"example_frames/{example_type}_{i}.png")
            if not image_path.exists():
                missing_frames.append(f"{example_type}_{i}.png")
        
        if missing_frames:
            missing_examples.append({
                "type": example_type,
                "missing": missing_frames,
                "complete": False
            })
        else:
            print(f"   ✅ {example_type}: Complete (4/4 images)")
    
    if missing_examples:
        print(f"\n⚠️  Missing example images for enhanced detection:")
        for example in missing_examples:
            print(f"   📝 {example['type']}: Missing {len(example['missing'])}/4 images")
            for missing_file in example['missing']:
                print(f"      - {missing_file}")
    
    return len(missing_examples) == 0

def print_deployment_checklist():
    """Print deployment checklist for Phase 2"""
    print("\n" + "="*60)
    print("🚀 Phase 2 Deployment Checklist")
    print("="*60)
    
    print("\n✅ COMPLETED:")
    print("   - Enhanced database.py with new fields")
    print("   - Enhanced action_detection.py with few-shot prompting")
    print("   - Updated app.py with operation type detection")
    print("   - Created prompt_examples.py for example management")
    
    print("\n📋 TODO:")
    print("   1. Add example images to example_frames/ directory:")
    print("      - manual_hanging_1.png through manual_hanging_4.png")
    print("      - forklift_hanging_1.png through forklift_hanging_4.png") 
    print("      - manual_removing_1.png through manual_removing_4.png")
    print("      - forklift_removing_1.png through forklift_removing_4.png")
    print("      - other_activity_1.png through other_activity_4.png")
    
    print("\n   2. Test enhanced detection:")
    print("      python action_detection.py test_image1.png test_image2.png test_image3.png test_image4.png")
    
    print("\n   3. Update dashboard.html to show operation types (Phase 4)")
    
    print("\n🔧 ENVIRONMENT VARIABLES:")
    print("   - USE_ENHANCED_DETECTION=true (default)")
    print("   - ENABLE_DELETION=false (recommended for testing)")
    print("   - LOG_LEVEL=DEBUG (for detailed testing)")

def print_example_guidance():
    """Print guidance on creating good example images"""
    print("\n" + "="*60)
    print("📸 Example Image Guidelines")
    print("="*60)
    
    guidelines = {
        "manual_hanging": [
            "Frame 1: Person approaching rack with parts in hand",
            "Frame 2: Person lifting/positioning parts",
            "Frame 3: Person placing parts on rack hooks/holders", 
            "Frame 4: Parts successfully hung, person stepping back"
        ],
        "forklift_hanging": [
            "Frame 1: Forklift approaching rack with parts on forks",
            "Frame 2: Operator positioning forklift for alignment",
            "Frame 3: Parts being placed on rack via forklift",
            "Frame 4: Parts hung successfully, forklift backing away"
        ],
        "manual_removing": [
            "Frame 1: Person approaching rack with hung parts",
            "Frame 2: Person reaching for/gripping parts on rack",
            "Frame 3: Person lifting parts off rack",
            "Frame 4: Person walking away with removed parts"
        ],
        "forklift_removing": [
            "Frame 1: Forklift positioning near rack with parts",
            "Frame 2: Forks being positioned under parts",
            "Frame 3: Parts being lifted off rack by forklift",
            "Frame 4: Forklift moving away with parts on forks"
        ],
        "other_activity": [
            "Frame 1: General activity (walking, maintenance, etc.)",
            "Frame 2: Continued non-hanging activity",
            "Frame 3: Workers doing other tasks near racks",
            "Frame 4: No parts being hung or removed from racks"
        ]
    }
    
    for activity, frames in guidelines.items():
        print(f"\n📋 {activity.replace('_', ' ').title()}:")
        for i, description in enumerate(frames, 1):
            print(f"   {i}. {description}")

def test_enhanced_detection():
    """Test if enhanced detection can be imported and run"""
    print("\n🧪 Testing enhanced detection imports...")
    
    try:
        from action_detection import validate_examples_setup, detect_rack_actions
        print("   ✅ action_detection imports successfully")
        
        validation = validate_examples_setup()
        if validation["examples_configured"]:
            print(f"   ✅ Examples configured: {validation['available_examples']}")
        else:
            print("   ⚠️  No examples configured yet")
        
        return True
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Validation error: {e}")
        return False

def main():
    print("🎯 Phase 2: Enhanced Action Detection Setup")
    print("=" * 50)
    
    # Step 1: Create directories
    create_directory_structure()
    
    # Step 2: Check example images
    examples_complete = check_example_images()
    
    # Step 3: Test imports
    imports_working = test_enhanced_detection()
    
    # Step 4: Print guidance
    print_deployment_checklist()
    
    if not examples_complete:
        print_example_guidance()
    
    # Summary
    print("\n" + "="*60)
    print("📊 PHASE 2 SETUP SUMMARY")
    print("="*60)
    print(f"   Directories: ✅ Ready")
    print(f"   Code imports: {'✅ Ready' if imports_working else '❌ Issues'}")
    print(f"   Example images: {'✅ Ready' if examples_complete else '⚠️  Needed'}")
    
    if examples_complete and imports_working:
        print("\n🎉 Phase 2 is ready to deploy!")
        print("   Start your app with: python app.py")
        print("   Check status at: http://127.0.0.1:5002/admin/enhanced-detection-status")
    else:
        print("\n📝 Next steps:")
        if not examples_complete:
            print("   1. Add example images to example_frames/ directory")
        if not imports_working:
            print("   2. Fix import issues with action_detection.py")
        print("   3. Re-run this setup script to verify")

if __name__ == "__main__":
    main()