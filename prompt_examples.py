"""
prompt_examples.py - Enhanced few-shot example library with variation improvements
--------------------------------------------------------------------------------
Manages example 4-frame sequences and their descriptions for training the LLM
to provide varied, specific responses instead of generic templates.

Key improvements:
- Multiple varied examples for each operation type
- Explicit variation instructions in prompts
- Chain-of-thought reasoning guidance
- Randomized example selection
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Directory containing example 4-frame sequences
EXAMPLES_DIR = Path("example_frames")

# Enhanced example descriptions with much more variation
EXAMPLE_DESCRIPTIONS = {
    "manual_hanging_1": {
        "operation_type": "manual_hanging",
        "description": "Worker carrying cylindrical tank components by hand to rack",
        "detailed_explanation": "A single worker manually lifts a large cylindrical metal tank. Frame 1 shows approach with tank in arms, Frame 2-3 show careful positioning and alignment with rack hooks, Frame 4 shows successful placement with worker stepping back to check alignment.",
        "key_details": "Heavy cylindrical parts, manual lifting technique, solo worker operation",
        "unique_aspects": "Worker uses proper lifting posture, tank requires precise angular alignment"
    },
    
    "manual_hanging_2": {
        "operation_type": "manual_hanging", 
        "description": "Two workers collaborating to hang structural beam sections",
        "detailed_explanation": "Two workers coordinate to lift and position long structural beams. Frame 1 shows workers approaching from both ends, Frame 2 shows synchronized lifting, Frame 3 shows careful positioning onto rack supports, Frame 4 shows both workers ensuring secure placement.",
        "key_details": "Long beam components, two-person teamwork, coordination required",
        "unique_aspects": "Requires communication between workers, beam length needs both ends supported"
    },
    
    "manual_hanging_3": {
        "operation_type": "manual_hanging",
        "description": "Worker using manual hoist to position heavy automotive part",
        "detailed_explanation": "Single worker operates a manual chain hoist to lift heavy automotive component. Frame 1 shows worker positioning hoist chain, Frame 2 shows lifting operation, Frame 3 shows careful guidance into rack position, Frame 4 shows securing and releasing hoist.",
        "key_details": "Chain hoist assistance, automotive parts, single operator with mechanical aid",
        "unique_aspects": "Mechanical assistance but still manual operation, requires hoist positioning skills"
    },
    
    "forklift_hanging_1": {
        "operation_type": "forklift_hanging",
        "description": "Standard forklift placing large cylindrical tanks using standard forks",
        "detailed_explanation": "Yellow industrial forklift carries multiple cylindrical tanks on standard forks. Frame 1 shows approach with tanks secured on forks, Frame 2 shows height adjustment and positioning, Frame 3 shows careful placement onto rack cradles, Frame 4 shows forklift backing away with empty forks.",
        "key_details": "Standard forklift forks, multiple cylindrical tanks, height adjustment critical",
        "unique_aspects": "Standard fork configuration, operator must judge rack clearance precisely"
    },
    
    "forklift_hanging_2": {
        "operation_type": "forklift_hanging",
        "description": "Forklift with spreader bar attachment handling wide structural components",
        "detailed_explanation": "Industrial forklift equipped with spreader bar attachment lifts wide structural panels. Frame 1 shows specialized attachment secured to wide components, Frame 2 shows approach with spotter guiding, Frame 3 shows precise positioning between rack posts, Frame 4 shows successful placement with spotter verification.",
        "key_details": "Spreader bar attachment, wide panels, spotter assistance, specialized equipment",
        "unique_aspects": "Custom attachment for wide loads, requires ground spotter for guidance"
    },
    
    "forklift_hanging_3": {
        "operation_type": "forklift_hanging", 
        "description": "Compact forklift maneuvering in tight spaces for lower rack placement",
        "detailed_explanation": "Small warehouse forklift navigates between rack posts to place boxed components. Frame 1 shows tight maneuvering between obstacles, Frame 2 shows low-height positioning, Frame 3 shows precise alignment with lower rack slots, Frame 4 shows backing out through narrow clearance.",
        "key_details": "Compact forklift, boxed components, tight clearances, lower rack levels",
        "unique_aspects": "Space constraints require precise maneuvering, lower rack placement technique"
    },
    
    # Additional variation examples for better training
    "manual_hanging_4": {
        "operation_type": "manual_hanging",
        "description": "Worker using rolling ladder to hang parts at elevated rack positions",
        "detailed_explanation": "Worker climbs rolling ladder while carrying smaller components to upper rack levels. Frame 1 shows ladder positioning and component preparation, Frame 2 shows climbing with parts, Frame 3 shows reaching and placing parts at height, Frame 4 shows descent and ladder repositioning.",
        "key_details": "Rolling ladder, elevated positioning, smaller parts, height safety",
        "unique_aspects": "Vertical access challenge, safety considerations for height work"
    },
    
    "forklift_hanging_4": {
        "operation_type": "forklift_hanging",
        "description": "Forklift with side-shift attachment for lateral positioning adjustments",
        "detailed_explanation": "Forklift equipped with side-shift hydraulics makes lateral adjustments for rack placement. Frame 1 shows initial positioning with side-shift visible, Frame 2 shows hydraulic lateral movement, Frame 3 shows fine-tuning alignment without moving entire forklift, Frame 4 shows final placement and retraction.",
        "key_details": "Side-shift hydraulics, lateral adjustment capability, precision placement",
        "unique_aspects": "Hydraulic side movement allows micro-adjustments without repositioning entire vehicle"
    }
}

# Area-specific prompt templates with variation emphasis
AREA_PROMPTS = {
    "general_labor": {
        "context": "general labor area where both manual workers and forklifts handle various parts and components",
        "focus": "Look for rack-related work activities including hanging parts, removing parts, moving racks, and loading operations",
        "common_operations": "manual hanging, forklift hanging, part organization, rack preparation"
    },
    
    "sandblast": {
        "context": "sandblast area where individual parts are prepared for and processed through sandblasting operations", 
        "focus": "Look for individual part handling activities including preparation, movement to/from booth, and post-processing",
        "common_operations": "part preparation, individual part handling, booth area activities"
    },
    
    "powder_booth": {
        "context": "powder coating booth area where parts undergo painting and coating processes",
        "focus": "Look for painting/coating activities and part movement in booth environment",
        "common_operations": "painting operations, coating application, booth activities"
    }
}

def get_available_examples() -> List[str]:
    """
    Get list of available example types (for backward compatibility).
    
    Returns:
        List of available example operation types
    """
    status = check_examples_directory()
    return status["available_examples"]

def check_examples_directory() -> Dict[str, any]:
    """
    Check which example images are available in the examples directory.
    
    Returns:
        Dictionary with availability info and recommendations
    """
    if not EXAMPLES_DIR.exists():
        return {
            "examples_configured": False,
            "available_examples": [],
            "missing_examples": list(EXAMPLE_DESCRIPTIONS.keys()),
            "recommendation": "Create example_frames/ directory and add example images"
        }
    
    available = []
    missing = []
    
    for example_name in EXAMPLE_DESCRIPTIONS.keys():
        # Check for the first frame of each example sequence
        example_file = EXAMPLES_DIR / f"{example_name}.png"
        if example_file.exists():
            available.append(example_name.split('_')[0] + '_' + example_name.split('_')[1])  # e.g., "manual_hanging"
        else:
            missing.append(example_name)
    
    # Remove duplicates from available (since we check multiple examples per type)
    available = list(set(available))
    
    return {
        "examples_configured": len(available) > 0,
        "available_examples": available,
        "missing_examples": missing,
        "recommendation": "Examples ready" if len(available) >= 2 else f"Add more examples: {missing}"
    }

def get_random_examples_by_type(operation_type: str, max_examples: int = 2) -> List[Dict]:
    """
    Get randomized examples for a specific operation type to improve variation.
    
    Args:
        operation_type: Type of operation (e.g., "manual_hanging", "forklift_hanging")
        max_examples: Maximum number of examples to return
        
    Returns:
        List of example dictionaries with randomized selection
    """
    # Filter examples by operation type
    type_examples = [
        (name, details) for name, details in EXAMPLE_DESCRIPTIONS.items()
        if details["operation_type"] == operation_type
    ]
    
    if not type_examples:
        return []
    
    # Randomize selection to avoid pattern repetition
    random.shuffle(type_examples)
    selected = type_examples[:max_examples]
    
    return [
        {
            "name": name,
            "description": details["description"],
            "explanation": details["detailed_explanation"],
            "key_details": details["key_details"],
            "unique_aspects": details["unique_aspects"]
        }
        for name, details in selected
    ]

def build_few_shot_prompt(area_type: str = "general_labor", max_examples: int = 2) -> str:
    """
    Build an enhanced few-shot prompt with explicit variation instructions.
    
    Args:
        area_type: Type of production area
        max_examples: Maximum examples per operation type
        
    Returns:
        Complete prompt string with examples and variation guidance
    """
    area_info = AREA_PROMPTS.get(area_type, AREA_PROMPTS["general_labor"])
    
    # Build the base prompt with strong variation emphasis
    prompt = f"""You are an industrial-vision assistant analyzing a 4-frame sequence in a {area_info['context']}.

🎯 CRITICAL INSTRUCTION: Provide SPECIFIC, UNIQUE explanations that vary based on what you actually observe.
DO NOT use generic template responses or repeat similar descriptions.
Focus on the PARTICULAR details that make THIS sequence different from others.

📋 Your task: {area_info['focus']}

Here are examples of different scenarios to guide your analysis style:

"""

    # Add varied examples for each available operation type
    available_types = ["manual_hanging", "forklift_hanging"]
    examples_added = 0
    
    for op_type in available_types:
        examples = get_random_examples_by_type(op_type, max_examples)
        
        if examples and examples_added < 4:  # Limit total examples to prevent prompt bloat
            prompt += f"\n--- {op_type.replace('_', ' ').upper()} EXAMPLES ---\n"
            
            for i, example in enumerate(examples[:2], 1):  # Max 2 per type
                prompt += f"""
Example {examples_added + 1}: {example['description']}
Detailed Analysis: "{example['explanation']}"
Key Details: {example['key_details']}
What Makes It Unique: {example['unique_aspects']}
"""
                examples_added += 1
                if examples_added >= 4:  # Total limit across all types
                    break
    
    # Add the analysis instructions with chain-of-thought guidance
    prompt += f"""

🔍 NOW ANALYZE THIS NEW 4-FRAME SEQUENCE:

Step 1: Observe each frame carefully
- Frame 1: What is the initial setup/approach?
- Frame 2: What movement/positioning occurs?
- Frame 3: What is the main action/placement?
- Frame 4: What is the completion/result?

Step 2: Identify unique characteristics
- What specific equipment/tools are visible?
- What type of parts are being handled (size, shape, material)?
- How many people are involved and what are their roles?
- What makes this operation distinct from the examples above?

Step 3: Determine operation classification
- Is this manual work, forklift operation, or something else?
- What is the confidence level based on clear visual evidence?
- Are there both manual and mechanical elements?

Step 4: Provide your specific analysis
Focus on details that make THIS sequence unique. Avoid generic descriptions.
Common operations in this area: {area_info['common_operations']}

Return ONLY JSON in this exact format:
{{
  "events": [
    {{
      "operation_type": "manual_hanging" | "forklift_hanging" | "manual_removing" | "forklift_removing" | "other",
      "explanation": "SPECIFIC description of THIS particular sequence with unique details observed",
      "confidence": 0.0-1.0,
      "forklift_detected": true/false,
      "manual_detected": true/false, 
      "people_count": number,
      "action": "specific action description based on what you see",
      "zone": "area description where activity occurs"
    }}
  ]
}}

🚨 REMEMBER: Be specific about what makes THIS operation unique. Avoid repeating template language from examples."""

    return prompt

def validate_examples_setup() -> Dict[str, any]:
    """
    Validate the current examples setup and provide recommendations.
    
    Returns:
        Dictionary with validation results and setup recommendations
    """
    status = check_examples_directory()
    
    # Enhanced validation with specific recommendations
    if not status["examples_configured"]:
        return {
            **status,
            "setup_quality": "Not configured",
            "variation_potential": "Low - no examples available",
            "recommendations": [
                "Create example_frames/ directory",
                "Add at least 2 manual_hanging examples (manual_hanging_1.png through manual_hanging_4.png)", 
                "Add at least 2 forklift_hanging examples (forklift_hanging_1.png through forklift_hanging_4.png)",
                "Use 4-frame sequences showing complete operations"
            ]
        }
    
    available_count = len(status["available_examples"])
    
    if available_count >= 2:
        variation_assessment = "High - multiple operation types with randomization"
        setup_quality = "Good"
        recommendations = [
            "Examples are working well",
            "Monitor for response variation in real usage",
            "Consider adding more examples if responses become repetitive"
        ]
    else:
        variation_assessment = "Medium - limited example variety"  
        setup_quality = "Basic"
        recommendations = [
            f"Add more example types beyond {status['available_examples']}",
            "Aim for at least 2 different operation types",
            "Add examples showing different scenarios (equipment, parts, techniques)"
        ]
    
    return {
        **status,
        "setup_quality": setup_quality,
        "variation_potential": variation_assessment,
        "recommendations": recommendations,
        "total_example_variations": len(EXAMPLE_DESCRIPTIONS),
        "randomization_enabled": True
    }

def get_example_status_summary() -> str:
    """
    Get a concise summary of example setup status for logging.
    
    Returns:
        Human-readable status string
    """
    status = validate_examples_setup()
    
    if not status["examples_configured"]:
        return "❌ No examples configured"
    
    available = status["available_examples"]
    setup_quality = status["setup_quality"]
    
    return f"✅ {len(available)} example types available ({', '.join(available)}) - Quality: {setup_quality}"

# Logging helper for debugging
def log_example_selection(selected_examples: List[str], operation_type: str = None):
    """Log which examples were selected for variation tracking."""
    if operation_type:
        logger.debug(f"Selected examples for {operation_type}: {selected_examples}")
    else:
        logger.debug(f"Selected examples: {selected_examples}")

# Export the main functions that other modules will use
__all__ = [
    'build_few_shot_prompt',
    'check_examples_directory', 
    'validate_examples_setup',
    'get_example_status_summary',
    'get_available_examples',  # Added for backward compatibility
    'EXAMPLE_DESCRIPTIONS',
    'AREA_PROMPTS'
]