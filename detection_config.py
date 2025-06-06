"""
detection_config.py - Configuration and tuning for action detection
"""

import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class DetectionConfig:
    """Configuration for action detection parameters"""
    
    # Confidence thresholds
    min_hanging_confidence: float = 0.85
    min_general_confidence: float = 0.70
    
    # Detection sensitivity
    use_two_stage_detection: bool = True
    require_upward_motion: bool = True
    require_proximity_to_rack: bool = True
    
    # Timing parameters
    min_sequence_length: int = 3  # Minimum frames needed
    max_sequence_length: int = 4  # Maximum frames to analyze
    
    # Area-specific settings
    area_configs: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.area_configs is None:
            self.area_configs = {
                "general_labor": {
                    "hanging_keywords": ["hanging", "placing", "reaching up"],
                    "exclude_keywords": ["walking", "standing idle", "moving rack"],
                    "min_confidence": 0.85,
                    "require_upward_motion": True,
                },
                "powder_booth": {
                    "action_keywords": ["painting", "spraying", "coating"],
                    "min_confidence": 0.75,
                    "require_upward_motion": False,
                },
                "sandblast": {
                    "action_keywords": ["preparing", "handling parts"],
                    "min_confidence": 0.70,
                    "require_upward_motion": False,
                }
            }

# Global configuration instance
CONFIG = DetectionConfig()

def update_config_from_env():
    """Update configuration from environment variables"""
    CONFIG.min_hanging_confidence = float(os.getenv("MIN_HANGING_CONFIDENCE", "0.85"))
    CONFIG.min_general_confidence = float(os.getenv("MIN_GENERAL_CONFIDENCE", "0.70"))
    CONFIG.use_two_stage_detection = os.getenv("USE_TWO_STAGE", "true").lower() == "true"
    CONFIG.require_upward_motion = os.getenv("REQUIRE_UPWARD_MOTION", "true").lower() == "true"

# Detailed prompts with explicit negative examples
ULTRA_SPECIFIC_PROMPTS = {
    "hanging_parts_detector": """
Analyze this 4-frame sequence to detect HANGING PARTS ON RACKS activity.

POSITIVE INDICATORS (report as "hanging parts"):
✓ Person reaching UP with arms above shoulder level toward rack hooks
✓ Deliberate placement motions of parts onto hanging points  
✓ Standing close to rack with clear upward reaching movements
✓ Parts being lifted and positioned on rack hangers/hooks

NEGATIVE INDICATORS (DO NOT report as hanging):
✗ Person walking past the rack without stopping
✗ Standing near rack but arms at sides or waist level
✗ Moving/pushing the entire rack structure  
✗ Picking up tools or materials from floor/table
✗ General conversation or supervision without part handling
✗ Adjusting already-hung parts (maintenance, not new hanging)

CRITICAL: Only count people actively performing the upward reaching/placement motion.

Return JSON with this exact structure:
{
  "events": [
    {
      "rack_id": "rack_01",
      "people_count": 1,
      "action": "hanging parts", 
      "confidence": 0.95,
      "zone": "rack_area",
      "validation": {
        "upward_reach_observed": true,
        "proximity_to_rack": true, 
        "placement_motion": true,
        "excluded_activities": []
      }
    }
  ]
}

If NO hanging activity is detected, return: {"events": []}
""",

    "movement_filter": """
First, determine if there is ANY significant human activity in this 4-frame sequence.

Look for:
- People moving or working
- Any form of interaction with equipment
- Changes in human positioning between frames

Return JSON:
{
  "activity_level": "high/medium/low/none",
  "people_present": 2,
  "brief_description": "Person working near rack area"
}

If activity_level is "none" or "low", subsequent hanging detection will be skipped.
""",

    "quality_assessment": """
Assess the quality and clarity of this 4-frame sequence for action detection:

Rate on these factors:
- Image clarity/resolution
- Lighting conditions  
- Person visibility
- Equipment/rack visibility
- Motion blur or artifacts

Return JSON:
{
  "overall_quality": "excellent/good/fair/poor",
  "clarity_score": 0.85,
  "recommended_confidence_adjustment": 0.0,
  "issues": ["motion blur in frame 3"]
}
"""
}

def get_adaptive_prompt(camera_id: str, detection_history: List[Dict] = None) -> str:
    """
    Generate adaptive prompt based on recent detection history
    """
    base_prompt = ULTRA_SPECIFIC_PROMPTS["hanging_parts_detector"]
    
    if not detection_history:
        return base_prompt
    
    # Analyze recent false positives/negatives
    recent_events = detection_history[-10:]  # Last 10 events
    false_positive_patterns = []
    
    for event in recent_events:
        # If this event was flagged as incorrect (would need feedback mechanism)
        if event.get("was_false_positive", False):
            false_positive_patterns.append(event.get("action", ""))
    
    if false_positive_patterns:
        additional_negatives = f"\n\nBased on recent corrections, ESPECIALLY avoid detecting these as hanging:\n"
        for pattern in set(false_positive_patterns):
            additional_negatives += f"✗ {pattern}\n"
        base_prompt += additional_negatives
    
    return base_prompt

def validate_detection_result(result: Dict, config: DetectionConfig) -> bool:
    """
    Post-process validation of detection results
    """
    if not result.get("events"):
        return True  # Empty results are valid
    
    for event in result["events"]:
        action = event.get("action", "").lower()
        confidence = event.get("confidence", 0)
        
        # Check confidence thresholds
        if "hanging" in action:
            if confidence < config.min_hanging_confidence:
                return False
        else:
            if confidence < config.min_general_confidence:
                return False
        
        # Check for required validation fields
        if config.require_upward_motion and "hanging" in action:
            validation = event.get("validation", {})
            if not validation.get("upward_reach_observed", False):
                return False
    
    return True

# Example usage configuration
EXAMPLE_ENV_VARS = """
# Add these to your environment for fine-tuning:

# Confidence thresholds (0.0 to 1.0)
export MIN_HANGING_CONFIDENCE=0.85
export MIN_GENERAL_CONFIDENCE=0.70

# Detection behavior
export USE_TWO_STAGE=true
export REQUIRE_UPWARD_MOTION=true

# LM Studio settings  
export LM_STUDIO_URL="http://127.0.0.1:1236/v1"
export ACTION_LM_MODEL="qwen/qwen2.5-vl-7b"

# Debugging
export LOG_LEVEL=DEBUG
"""

if __name__ == "__main__":
    print("Detection Configuration")
    print("=" * 50)
    print(f"Min hanging confidence: {CONFIG.min_hanging_confidence}")
    print(f"Use two-stage detection: {CONFIG.use_two_stage_detection}")
    print(f"Require upward motion: {CONFIG.require_upward_motion}")
    print("\nArea configs:")
    for area, conf in CONFIG.area_configs.items():
        print(f"  {area}: min_conf={conf['min_confidence']}")
    
    print("\nEnvironment variables for tuning:")
    print(EXAMPLE_ENV_VARS)