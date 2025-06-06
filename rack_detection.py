"""
rack_detection.py – Enhanced rack detection with global position-based tracking
------------------------------------------------------------------------------
Sends four chronological frames to LM-Studio with position-based context.
Integrates with global rack state management for consistent ID assignment.
Handles cross-zone movement detection and rack identity persistence.
"""

import base64, json, logging, os, re, requests, time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

# Import our tracking and state management
from llm_tracking import llm_logger, log_llm_request, log_llm_response, log_parsing_attempt, log_error_details
from rack_state_manager import rack_state_manager
from rack_transitions import rack_transition_manager
from rack_prompt_examples import build_position_based_prompt, get_rack_examples_status

logger = logging.getLogger(__name__)

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1236/v1")
MODEL_NAME    = os.getenv("RACK_LM_MODEL", "qwen/qwen2.5-vl-7b")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

def _b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()

def _strip_fence(txt: str) -> str:
    m = _FENCE_RE.search(txt)
    return m.group(1).strip() if m else txt.strip()

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _parse_enhanced_rack_response(response_text: str, camera_id: str) -> List[Dict]:
    """
    Parse the enhanced rack detection response with position validation.
    """
    log_parsing_attempt("rack_detection", response_text, None, "initial_json")
    
    def fix_incomplete_json(text):
        llm_logger.debug("Attempting to fix incomplete rack detection JSON")
        text = text.strip()
        
        # Handle common JSON issues
        open_braces = text.count('{')
        close_braces = text.count('}')
        
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            text += '}' * missing_braces
            llm_logger.debug(f"Added {missing_braces} missing closing braces")
        
        if text.endswith(','):
            text = text.rstrip(',')
            llm_logger.debug("Removed trailing comma")
        
        return text
    
    try:
        data = json.loads(response_text)
        
        # Handle both formats: {"racks": [...]} and [...]
        if isinstance(data, list):
            racks = data  # Direct list format
            llm_logger.info("LLM returned direct list format")
        elif isinstance(data, dict) and "racks" in data:
            racks = data["racks"]  # Wrapped format
            llm_logger.info("LLM returned wrapped format")
        else:
            llm_logger.warning("Unexpected JSON format from rack detection LLM")
            racks = []
        
        log_parsing_attempt("rack_detection", response_text, racks, "successful_json")
        
    except json.JSONDecodeError as e:
        llm_logger.warning(f"Initial rack detection JSON parse failed: {e}")
        try:
            # Try to fix the JSON
            fixed_json = fix_incomplete_json(response_text)
            llm_logger.debug(f"Attempting to parse fixed JSON: {fixed_json[:200]}")
            data = json.loads(fixed_json)
            
            # Handle both formats for fixed JSON too
            if isinstance(data, list):
                racks = data
            elif isinstance(data, dict) and "racks" in data:
                racks = data["racks"]
            else:
                racks = []
            
            log_parsing_attempt("rack_detection", fixed_json, racks, "fixed_json")
            llm_logger.info("Successfully parsed fixed rack detection JSON")
        except json.JSONDecodeError as e2:
            llm_logger.error(f"Even fixed rack detection JSON failed to parse: {e2}")
            log_parsing_attempt("rack_detection", fixed_json if 'fixed_json' in locals() else response_text, None, "failed_fixed_json")
            
            # Last resort: try to extract partial information
            racks = _extract_partial_rack_response(response_text)
            log_parsing_attempt("rack_detection", response_text, racks, "partial_extraction")
    
    # Process and normalize the rack data
    normalized_racks = []
    ts = _now_iso()
    
    for i, rack in enumerate(racks):
        llm_logger.debug(f"Processing rack {i}: {rack}")
        
        # Validate and normalize rack fields
        rack_id = rack.get("id", f"rack_{i+1:02d}")
        zone = rack.get("zone", "unknown")
        moving = rack.get("moving", False)
        confidence = min(1.0, max(0.0, float(rack.get("confidence", 0.8))))
        position_description = rack.get("position_description", f"{zone} area")
        
        normalized_rack = {
            "timestamp": ts,
            "camera_id": camera_id,
            "rack_id": rack_id,
            "global_rack_id": None,  # Will be assigned by state manager
            "x": None,  # VLM doesn't provide coordinates
            "y": None,
            "moving": 1 if moving else 0,
            "zone_description": zone,
            "confidence": confidence,
            "position_description": position_description
        }
        
        normalized_racks.append(normalized_rack)
        llm_logger.debug(f"Normalized rack {i}: {normalized_rack}")
    
    llm_logger.info(f"Successfully parsed {len(normalized_racks)} racks for {camera_id}")
    return normalized_racks

def _extract_partial_rack_response(text: str) -> List[Dict]:
    """Extract rack information even from malformed JSON response."""
    llm_logger.info("Attempting partial extraction from malformed rack response")
    
    racks = []
    
    # Look for rack patterns in the text
    rack_patterns = [
        r'"id":\s*"(rack_\w+)"',
        r'"zone":\s*"([^"]+)"',
        r'"moving":\s*(true|false)',
        r'"confidence":\s*([0-9.]+)'
    ]
    
    # Try to extract basic rack information
    rack_ids = re.findall(rack_patterns[0], text, re.IGNORECASE)
    zones = re.findall(rack_patterns[1], text, re.IGNORECASE)
    moving_flags = re.findall(rack_patterns[2], text, re.IGNORECASE)
    confidences = re.findall(rack_patterns[3], text, re.IGNORECASE)
    
    # Create racks from extracted data
    max_racks = max(len(rack_ids), len(zones), 1)
    
    for i in range(max_racks):
        rack_id = rack_ids[i] if i < len(rack_ids) else f"rack_{i+1:02d}"
        zone = zones[i] if i < len(zones) else "unknown"
        moving = moving_flags[i].lower() == 'true' if i < len(moving_flags) else False
        confidence = float(confidences[i]) if i < len(confidences) else 0.7
        
        rack = {
            "timestamp": _now_iso(),
            "camera_id": "",  # Will be filled by caller
            "rack_id": rack_id,
            "global_rack_id": None,
            "x": None,
            "y": None,
            "moving": 1 if moving else 0,
            "zone_description": zone,
            "confidence": confidence,
            "position_description": f"{zone} area"
        }
        
        racks.append(rack)
    
    llm_logger.info(f"Partial extraction successful: {len(racks)} racks")
    return racks

def detect_racks(image_paths: List[str], 
                camera_id: str,
                timeout: int = 90) -> List[Dict]:
    """
    Enhanced rack detection with global state management and position-based tracking.
    
    Args:
        image_paths: List of 4 image file paths
        camera_id: Camera identifier ("general_labor" or "powder_booth")
        timeout: Request timeout in seconds
        
    Returns:
        List of rack position dictionaries with global IDs assigned
    """
    if len(image_paths) != 4:
        raise ValueError("detect_racks expects exactly 4 images")

    llm_logger.info(f"=== STARTING ENHANCED RACK DETECTION ===")
    llm_logger.info(f"Camera: {camera_id}")
    llm_logger.info(f"Images: {len(image_paths)}")
    llm_logger.info(f"Timeout: {timeout}s")
    
    # Log image paths and their existence
    for i, path in enumerate(image_paths):
        exists = Path(path).exists()
        size = Path(path).stat().st_size if exists else 0
        llm_logger.info(f"Image {i+1}: {path} (exists: {exists}, size: {size} bytes)")

    logger.info("Enhanced rack detection: analyzing 4-frame burst for %s", camera_id)
    
    # Get current rack state context for this camera
    rack_context = rack_state_manager.build_prompt_context(camera_id)
    llm_logger.info(f"Built rack context for {camera_id}")
    llm_logger.debug(f"Rack context: {rack_context}")
    
    # Build position-based prompt with context
    prompt = build_position_based_prompt(camera_id, rack_context)
    llm_logger.info(f"Using position-based prompt for {camera_id}")
    
    # Log the prompt being used (truncated for readability)
    prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
    llm_logger.info(f"Prompt preview: {prompt_preview}")
    
def add_reference_images_to_content(content: List[Dict], camera_id: str) -> List[Dict]:
    """
    Add reference rack images to the prompt content to help LLM distinguish racks from other equipment.
    
    Args:
        content: Existing content list with text prompt
        camera_id: Camera identifier for context
        
    Returns:
        Updated content list with reference images
    """
    # Define reference image paths
    reference_dir = Path("rack_references")
    
    # Look for reference images
    reference_images = []
    
    # Try to find camera-specific references first
    camera_ref_dir = reference_dir / camera_id
    if camera_ref_dir.exists():
        reference_images.extend(sorted(camera_ref_dir.glob("*.png"))[:2])  # Max 2 per camera
        reference_images.extend(sorted(camera_ref_dir.glob("*.jpg"))[:2])
    
    # Fall back to general references
    if not reference_images and reference_dir.exists():
        reference_images.extend(sorted(reference_dir.glob("rack_example_*.png"))[:2])
        reference_images.extend(sorted(reference_dir.glob("rack_example_*.jpg"))[:2])
    
    if reference_images:
        # Add reference images at the beginning (after text prompt but before analysis images)
        reference_content = []
        
        # Add explanation text for reference images
        reference_content.append({
            "type": "text",
            "text": f"\n🔍 REFERENCE IMAGES - These show examples of RACKS you should detect:\n"
        })
        
        # Add reference images
        for i, ref_img in enumerate(reference_images[:2], 1):  # Limit to 2 reference images
            try:
                ref_b64 = _b64(ref_img)
                reference_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{ref_b64}"}
                })
                llm_logger.info(f"Added reference image {i}: {ref_img.name}")
            except Exception as e:
                llm_logger.warning(f"Failed to load reference image {ref_img}: {e}")
        
        reference_content.append({
            "type": "text", 
            "text": f"\n📋 NOW ANALYZE these 4 frames from {camera_id} and identify SIMILAR RACK STRUCTURES:\n"
        })
        
        # Insert reference content after the main prompt but before analysis images
        # content[0] is the main text prompt
        content = [content[0]] + reference_content + content[1:]
        
        llm_logger.info(f"Added {len([c for c in reference_content if c['type'] == 'image_url'])} reference images")
    
    else:
        llm_logger.warning("No reference images found - LLM may confuse racks with other equipment")
        llm_logger.info("💡 Add reference images to: rack_references/rack_example_1.png, rack_references/rack_example_2.png")
        llm_logger.info("💡 Or camera-specific: rack_references/general_labor/rack_1.png, rack_references/powder_booth/rack_1.png")
    
    return content
    
    for p in map(Path, image_paths):
        if not p.exists():
            llm_logger.error(f"Image file not found: {p}")
            logger.warning("Image file not found: %s", p)
            continue
            
        try:
            image_b64 = _b64(p)
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            })
            valid_images += 1
            llm_logger.debug(f"Added image to payload: {p} (base64 length: {len(image_b64)})")
        except Exception as e:
            llm_logger.error(f"Failed to encode image {p}: {e}")
            log_error_details("rack_detection", e, {"image_path": str(p)})

    if len(content) == 1:  # Only text, no valid images
        llm_logger.error("No valid images found for rack analysis")
        logger.error("No valid images found for rack analysis")
        return []

    total_images = valid_images + reference_images_added
    llm_logger.info(f"Successfully prepared {valid_images} analysis images + {reference_images_added} reference images = {total_images} total")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 1500,
    }

    # Log the full request AND save the images
    log_llm_request("rack_detection", payload, total_images, camera_id)

    url = (f"{LM_STUDIO_URL}/chat/completions"
           if not LM_STUDIO_URL.endswith("/chat/completions")
           else LM_STUDIO_URL)

    try:
        llm_logger.info(f"Sending request to LM-Studio: {url}")
        logger.debug("Sending rack detection request to LM-Studio with %d images", len(content) - 1)
        
        start_time = time.time()
        r = requests.post(url, json=payload, timeout=timeout)
        processing_time = time.time() - start_time
        
        llm_logger.info(f"Request completed in {processing_time:.2f}s with status {r.status_code}")
        
        r.raise_for_status()
    except Exception as exc:
        log_error_details("rack_detection", exc, {
            "url": url,
            "timeout": timeout,
            "image_count": valid_images
        })
        logger.exception("Enhanced rack detection: HTTP failure – %s", exc)
        return []

    # Parse response
    try:
        body = r.json()
        log_llm_response("rack_detection", body, body.get("choices", [{}])[0].get("message", {}).get("content", ""), processing_time)
    except ValueError as e:
        llm_logger.error(f"Failed to parse response as JSON: {e}")
        llm_logger.error(f"Raw response text: {r.text[:1000]}")
        logger.error("Enhanced rack detection: plain-text LM response: %.150s …",
                     r.text.replace('\\n', ' '))
        return []

    if "choices" not in body:
        llm_logger.error(f"No 'choices' in response body: {json.dumps(body)[:400]}")
        logger.error("Enhanced rack detection: LM error – %s", json.dumps(body)[:400])
        return []

    raw_response = body["choices"][0]["message"]["content"]
    clean_response = _strip_fence(raw_response)
    
    llm_logger.info(f"Raw LM response: {raw_response}")
    llm_logger.info(f"Cleaned response: {clean_response}")
    logger.debug("Enhanced rack detection: raw LM content for %s: %.300s", camera_id, clean_response)

    # Parse with enhanced parser
    detected_racks = _parse_enhanced_rack_response(clean_response, camera_id)
    
    if not detected_racks:
        llm_logger.info(f"No racks detected in {camera_id}")
        logger.debug("Enhanced rack detection: no racks detected in %s", camera_id)
        llm_logger.info(f"=== ENHANCED RACK DETECTION COMPLETE (NO RACKS) ===")
        return []
    
    # Assign global rack IDs using state manager
    try:
        racks_with_global_ids = rack_state_manager.assign_rack_ids(detected_racks, camera_id)
        llm_logger.info(f"Assigned global IDs to {len(racks_with_global_ids)} racks")
        
        for rack in racks_with_global_ids:
            logger.info("  - %s (%s) in %s: moving=%s, confidence=%.2f", 
                       rack["rack_id"], rack["global_rack_id"], 
                       rack["zone_description"], bool(rack["moving"]), rack["confidence"])
    except Exception as e:
        llm_logger.error(f"Failed to assign global rack IDs: {e}")
        log_error_details("rack_detection", e, {"camera_id": camera_id, "detected_racks": len(detected_racks)})
        # Fall back to detected racks without global IDs
        racks_with_global_ids = detected_racks
    
    logger.info("Enhanced rack detection: detected %d racks in %s", len(racks_with_global_ids), camera_id)
    llm_logger.info(f"=== ENHANCED RACK DETECTION COMPLETE ===")
    
    return racks_with_global_ids

def detect_cross_zone_movements(general_labor_racks: List[Dict], 
                               powder_booth_racks: List[Dict]) -> List[Dict]:
    """
    Detect rack movements between general_labor and powder_booth zones.
    
    Args:
        general_labor_racks: Racks detected in general_labor camera
        powder_booth_racks: Racks detected in powder_booth camera
        
    Returns:
        List of detected movement events
    """
    try:
        llm_logger.info("=== DETECTING CROSS-ZONE MOVEMENTS ===")
        llm_logger.info(f"General labor racks: {len(general_labor_racks)}")
        llm_logger.info(f"Powder booth racks: {len(powder_booth_racks)}")
        
        movements = rack_transition_manager.detect_cross_zone_movements(
            general_labor_racks, powder_booth_racks
        )
        
        if movements:
            llm_logger.info(f"Detected {len(movements)} cross-zone movements")
            # Process the movements
            rack_transition_manager.process_detected_movements(movements)
        else:
            llm_logger.info("No cross-zone movements detected")
        
        # Resolve any pending transitions
        resolved_count = rack_transition_manager.resolve_pending_transitions()
        if resolved_count > 0:
            llm_logger.info(f"Resolved {resolved_count} pending transitions")
        
        return movements
        
    except Exception as e:
        logger.error(f"Failed to detect cross-zone movements: {e}")
        log_error_details("rack_transitions", e, {
            "gl_racks": len(general_labor_racks),
            "pb_racks": len(powder_booth_racks)
        })
        return []

def validate_rack_detection_setup() -> Dict:
    """
    Validate that enhanced rack detection is properly configured.
    
    Returns:
        Dictionary with validation results and recommendations
    """
    try:
        # Check examples setup
        examples_status = get_rack_examples_status()
        
        # Check state manager
        validation = rack_state_manager.validate_rack_assignments("general_labor")
        gl_validation = validation
        
        validation = rack_state_manager.validate_rack_assignments("powder_booth")
        pb_validation = validation
        
        # Check transition manager
        transition_summary = rack_transition_manager.get_transition_summary()
        
        total_issues = gl_validation.get("total_issues", 0) + pb_validation.get("total_issues", 0)
        
        return {
            "setup_status": "configured" if "✅" in examples_status else "needs_setup",
            "examples_status": examples_status,
            "general_labor_validation": gl_validation,
            "powder_booth_validation": pb_validation,
            "transition_summary": transition_summary,
            "total_issues": total_issues,
            "recommendation": "Rack detection ready" if total_issues == 0 else "Review rack tracking issues"
        }
        
    except Exception as e:
        logger.error(f"Failed to validate rack detection setup: {e}")
        return {
            "setup_status": "error",
            "examples_status": "❌ Validation failed",
            "total_issues": 1,
            "error": str(e),
            "recommendation": "Check rack detection system configuration"
        }

def cleanup_old_rack_data(hours_threshold: int = 24) -> Dict:
    """
    Clean up old rack tracking data and resolve stale states.
    
    Args:
        hours_threshold: Hours after which to consider data stale
        
    Returns:
        Dictionary with cleanup results
    """
    try:
        llm_logger.info(f"=== CLEANING UP RACK DATA (threshold: {hours_threshold}h) ===")
        
        # Cleanup stale racks
        cleaned_racks = rack_state_manager.cleanup_stale_racks(hours_threshold)
        
        # Resolve pending transitions
        resolved_transitions = rack_transition_manager.resolve_pending_transitions()
        
        # Validate after cleanup
        gl_validation = rack_state_manager.validate_rack_assignments("general_labor")
        pb_validation = rack_state_manager.validate_rack_assignments("powder_booth")
        
        cleanup_results = {
            "timestamp": _now_iso(),
            "hours_threshold": hours_threshold,
            "cleaned_stale_racks": cleaned_racks,
            "resolved_transitions": resolved_transitions,
            "post_cleanup_validation": {
                "general_labor": gl_validation,
                "powder_booth": pb_validation
            }
        }
        
        llm_logger.info(f"Cleanup complete: {cleaned_racks} stale racks, {resolved_transitions} transitions resolved")
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Failed to cleanup rack data: {e}")
        return {
            "timestamp": _now_iso(),
            "error": str(e),
            "cleaned_stale_racks": 0,
            "resolved_transitions": 0
        }

def get_rack_detection_summary() -> Dict:
    """
    Get a comprehensive summary of rack detection status.
    
    Returns:
        Dictionary with current rack detection state
    """
    try:
        # Get current rack states
        gl_context = rack_state_manager.get_current_rack_context("general_labor")
        pb_context = rack_state_manager.get_current_rack_context("powder_booth")
        
        # Get transition summary
        transition_summary = rack_transition_manager.get_transition_summary()
        
        # Get validation status
        validation = validate_rack_detection_setup()
        
        return {
            "timestamp": _now_iso(),
            "general_labor": {
                "total_racks": gl_context["total_active_racks"],
                "current_racks": gl_context["current_racks"],
                "recent_departures": len(gl_context["recently_departed"]),
                "recent_arrivals": len(gl_context["recently_arrived"])
            },
            "powder_booth": {
                "total_racks": pb_context["total_active_racks"], 
                "current_racks": pb_context["current_racks"],
                "recent_departures": len(pb_context["recently_departed"]),
                "recent_arrivals": len(pb_context["recently_arrived"])
            },
            "transitions": transition_summary,
            "validation": validation,
            "system_status": "operational" if validation["total_issues"] == 0 else "issues_detected"
        }
        
    except Exception as e:
        logger.error(f"Failed to get rack detection summary: {e}")
        return {
            "timestamp": _now_iso(),
            "error": str(e),
            "system_status": "error"
        }

# Export main functions
__all__ = [
    'detect_racks',
    'detect_cross_zone_movements', 
    'validate_rack_detection_setup',
    'cleanup_old_rack_data',
    'get_rack_detection_summary'
]