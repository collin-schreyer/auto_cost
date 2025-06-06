"""
rack_prompt_examples.py - Position-based rack detection examples and prompt building
----------------------------------------------------------------------------------
Provides few-shot examples and prompts focused on position-based rack identification
rather than activity detection. Emphasizes consistent ID assignment across time.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Directory containing example rack detection sequences
EXAMPLES_DIR = Path("rack_examples")

# Position-based rack detection examples
RACK_DETECTION_EXAMPLES = {
    "general_labor_stable": {
        "camera": "general_labor",
        "scenario": "Multiple racks in stable positions",
        "description": "Three racks in consistent positions across 4 frames",
        "detailed_explanation": "Frame sequence shows rack_01 consistently in left_side zone, rack_02 in center_left zone, and rack_03 in right_side zone. No movement detected - racks maintain positions.",
        "rack_positions": {
            "rack_01": "left_side - against wall, 4-shelf metal unit",
            "rack_02": "center_left - mobile rack with wheels", 
            "rack_03": "right_side - near conveyor, tall frame"
        },
        "expected_output": {
            "racks": [
                {"id": "rack_01", "zone": "left_side", "moving": False, "confidence": 0.9},
                {"id": "rack_02", "zone": "center_left", "moving": False, "confidence": 0.9},
                {"id": "rack_03", "zone": "right_side", "moving": False, "confidence": 0.8}
            ]
        }
    },
    
    "general_labor_movement": {
        "camera": "general_labor", 
        "scenario": "One rack moving between zones",
        "description": "Two stable racks, one rack moving from center to right side",
        "detailed_explanation": "Frame 1-2: rack_02 visible in center_left zone. Frame 3-4: rack_02 appears in center_right zone, indicating movement. rack_01 and rack_03 remain stable in their positions.",
        "rack_positions": {
            "rack_01": "left_side - stable position throughout",
            "rack_02": "center_left → center_right - clear movement detected",
            "rack_03": "right_side - stable position throughout"  
        },
        "expected_output": {
            "racks": [
                {"id": "rack_01", "zone": "left_side", "moving": False, "confidence": 0.9},
                {"id": "rack_02", "zone": "center_right", "moving": True, "confidence": 0.8},
                {"id": "rack_03", "zone": "right_side", "moving": False, "confidence": 0.9}
            ]
        }
    },
    
    "powder_booth_stable": {
        "camera": "powder_booth",
        "scenario": "Racks in powder coating area",
        "description": "Two racks positioned for coating operations",
        "detailed_explanation": "Frame sequence shows rack_01 at entrance zone (parts delivery position) and rack_02 at center zone (active coating position). Both racks stationary throughout sequence.",
        "rack_positions": {
            "rack_01": "entrance - parts staging area",
            "rack_02": "center - coating position"
        },
        "expected_output": {
            "racks": [
                {"id": "rack_01", "zone": "entrance", "moving": False, "confidence": 0.9},
                {"id": "rack_02", "zone": "center", "moving": False, "confidence": 0.9}
            ]
        }
    },
    
    "transition_scenario": {
        "camera": "general_labor",
        "scenario": "Rack entering from other zone",
        "description": "New rack appears in transition zone",
        "detailed_explanation": "Frame 1-2: Two existing racks (rack_01, rack_02) in stable positions. Frame 3-4: New rack appears in right_side transition zone, likely arrived from powder_booth area.",
        "rack_positions": {
            "rack_01": "left_side - existing stable rack",
            "rack_02": "center_left - existing stable rack", 
            "rack_03": "right_side - newly appeared, transition zone"
        },
        "expected_output": {
            "racks": [
                {"id": "rack_01", "zone": "left_side", "moving": False, "confidence": 0.9},
                {"id": "rack_02", "zone": "center_left", "moving": False, "confidence": 0.9},
                {"id": "rack_03", "zone": "right_side", "moving": False, "confidence": 0.7}
            ]
        }
    }
}

# Zone configuration for cameras
CAMERA_ZONES = {
    "general_labor": {
        "zones": ["left_side", "center_left", "center_right", "right_side"],
        "transition_zones": ["right_side"],
        "description": "General labor area with 4 main zones. Racks typically enter from left_side and exit via right_side to powder_booth."
    },
    "powder_booth": {
        "zones": ["entrance", "center", "back_wall"],
        "transition_zones": ["entrance"],
        "description": "Powder coating area with 3 zones. Racks enter via entrance zone and may be positioned at center or back_wall for coating."
    }
}

def get_camera_zone_description(camera_id: str) -> str:
    """Get zone layout description for a camera."""
    zone_info = CAMERA_ZONES.get(camera_id, {})
    zones = zone_info.get("zones", [])
    transition_zones = zone_info.get("transition_zones", [])
    description = zone_info.get("description", f"Camera {camera_id} zone layout")
    
    zone_text = f"Zone layout: {', '.join(zones)}"
    if transition_zones:
        zone_text += f"\nTransition zones (racks may move to/from other cameras): {', '.join(transition_zones)}"
    
    return f"{description}\n{zone_text}"

def get_position_examples_for_camera(camera_id: str, max_examples: int = 2) -> List[Dict]:
    """
    Get position-based examples specific to a camera type.
    
    Args:
        camera_id: Camera identifier ("general_labor" or "powder_booth")
        max_examples: Maximum number of examples to return
        
    Returns:
        List of relevant examples for the camera
    """
    # Filter examples by camera
    relevant_examples = [
        (name, details) for name, details in RACK_DETECTION_EXAMPLES.items()
        if details["camera"] == camera_id or camera_id in name
    ]
    
    # If no camera-specific examples, include general examples
    if not relevant_examples:
        relevant_examples = list(RACK_DETECTION_EXAMPLES.items())
    
    # Randomize to avoid pattern repetition
    random.shuffle(relevant_examples)
    selected = relevant_examples[:max_examples]
    
    return [
        {
            "name": name,
            "scenario": details["scenario"],
            "description": details["description"],
            "explanation": details["detailed_explanation"],
            "positions": details["rack_positions"],
            "expected_output": details["expected_output"]
        }
        for name, details in selected
    ]

def build_position_based_prompt(camera_id: str, rack_context: str = "") -> str:
    """
    Build a position-based rack detection prompt with context and examples.
    
    Args:
        camera_id: Camera identifier
        rack_context: Current rack state context from rack_state_manager
        
    Returns:
        Complete prompt for position-based rack detection
    """
    # Get zone information
    zone_description = get_camera_zone_description(camera_id)
    
    # Get relevant examples
    examples = get_position_examples_for_camera(camera_id, max_examples=2)
    
    prompt = f"""You are an industrial rack tracking assistant analyzing a 4-frame sequence from {camera_id} camera.

🎯 OBJECTIVE: Identify RACKS by their POSITIONS and assign consistent IDs based on their locations in the workspace.

⚠️ CRITICAL: ONLY detect RACKS, not other equipment:
• RACKS are stationary metal structures with horizontal shelves/bars for hanging parts
• RACKS are tall, vertical structures that hold parts or components
• DO NOT confuse with forklifts, vehicles, people, or other mobile equipment
• FORKLIFTS are vehicles with forks - these are NOT racks
• EQUIPMENT or MACHINERY are not racks

🏗️ WHAT RACKS LOOK LIKE:
• Vertical metal frame structures
• Multiple horizontal levels/shelves
• Used for storing or hanging parts
• Usually stationary (not vehicles)
• May have parts hanging from them
• Often rectangular or linear in shape

📍 ZONE LAYOUT FOR {camera_id.upper()}:
{zone_description}

{rack_context}

🔍 ANALYSIS INSTRUCTIONS:

1. **Position-Based Identification**: 
   - Assign rack IDs based on their zone positions (left_side = rack_01, center = rack_02, etc.)
   - Maintain consistent IDs for racks in the same positions across time
   - If a rack moves zones, track it to its new position

2. **Movement Detection**:
   - Compare rack positions across frames 1→2→3→4
   - Mark "moving: true" only if rack clearly changes zones between frames
   - Static racks in same zone = "moving: false"

3. **Zone Assignment**:
   - Identify which zone each rack occupies in the final frame
   - Use zone names: {', '.join(CAMERA_ZONES.get(camera_id, {}).get('zones', ['left', 'center', 'right']))}

Here are examples of position-based rack analysis:

"""

    # Add examples
    for i, example in enumerate(examples, 1):
        prompt += f"""
--- EXAMPLE {i}: {example['scenario']} ---
Scenario: {example['description']}
Analysis: "{example['explanation']}"
Rack Positions: {example['positions']}
Expected Output: {example['expected_output']}
"""

    prompt += f"""

🔍 NOW ANALYZE THIS 4-FRAME SEQUENCE:

Step 1: Identify rack positions in each frame
- Frame 1: What racks are visible and in which zones?
- Frame 2: Any position changes from Frame 1?
- Frame 3: Any position changes from Frame 2?  
- Frame 4: Final positions of all racks?

Step 2: Assign position-based IDs
- leftmost rack = rack_01, next = rack_02, etc.
- Maintain consistent IDs for racks staying in same zones
- Update IDs if racks move to different zones

Step 3: Detect movements
- Did any rack change zones between frames?
- Only mark as "moving: true" if zone change is detected

Step 4: Assess confidence
- High confidence (0.9): Clear rack in obvious zone position
- Medium confidence (0.7-0.8): Rack visible but position partially unclear
- Low confidence (0.5-0.6): Rack barely visible or zone ambiguous

Return ONLY JSON in this exact format:
{{
  "racks": [
    {{
      "id": "rack_01",
      "zone": "left_side",
      "moving": false,
      "confidence": 0.9,
      "position_description": "against left wall, 4-shelf unit"
    }}
  ]
}}

🚨 CRITICAL: Focus on POSITION-based identification. Racks in the same zones should get consistent IDs over time.
"""

    return prompt

def validate_rack_examples_setup() -> Dict:
    """
    Validate rack detection examples setup.
    
    Returns:
        Dictionary with validation results
    """
    if not EXAMPLES_DIR.exists():
        return {
            "examples_configured": False,
            "available_examples": [],
            "missing_examples": list(RACK_DETECTION_EXAMPLES.keys()),
            "recommendation": "Create rack_examples/ directory and add example images"
        }
    
    available = []
    missing = []
    
    for example_name in RACK_DETECTION_EXAMPLES.keys():
        # Check for example files (looking for frame sequences)
        example_files = list(EXAMPLES_DIR.glob(f"{example_name}_*.png"))
        if len(example_files) >= 4:  # Need at least 4 frames
            available.append(example_name)
        else:
            missing.append(example_name)
    
    return {
        "examples_configured": len(available) > 0,
        "available_examples": available,
        "missing_examples": missing,
        "total_example_scenarios": len(RACK_DETECTION_EXAMPLES),
        "recommendation": "Examples ready" if len(available) >= 2 else f"Add more examples: {missing[:3]}..."
    }

def get_rack_examples_status() -> str:
    """
    Get a concise status summary of rack detection examples.
    
    Returns:
        Human-readable status string
    """
    status = validate_rack_examples_setup()
    
    if not status["examples_configured"]:
        return "❌ No rack detection examples configured"
    
    available = len(status["available_examples"])
    total = status["total_example_scenarios"]
    
    return f"✅ {available}/{total} rack detection examples available"

def log_prompt_selection(camera_id: str, selected_examples: List[str]):
    """Log which examples were selected for debugging."""
    logger.debug(f"Rack detection prompt for {camera_id}: selected examples {selected_examples}")

# Export main functions
__all__ = [
    'build_position_based_prompt',
    'get_camera_zone_description', 
    'validate_rack_examples_setup',
    'get_rack_examples_status',
    'get_position_examples_for_camera',
    'RACK_DETECTION_EXAMPLES',
    'CAMERA_ZONES'
]