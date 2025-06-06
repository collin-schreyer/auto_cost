"""
Enhanced action_detection.py with area-specific detection rules and FIXED JSON parsing
"""
import base64, json, logging, os, re, requests, time
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timezone

# Import our enhanced tracking with image support
from llm_tracking import llm_logger, log_llm_request, log_llm_response, log_parsing_attempt, log_error_details

# Import our example library
from prompt_examples import build_few_shot_prompt, get_available_examples, check_examples_directory, validate_examples_setup

logger        = logging.getLogger(__name__)
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1236/v1")
MODEL_NAME    = os.getenv("ACTION_LM_MODEL", "qwen/qwen2.5-vl-7b")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

def _b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()

def _strip_fence(txt: str) -> str:
    """Extract JSON from markdown code fences, or return original text if no fences found."""
    m = _FENCE_RE.search(txt)
    if m:
        return m.group(1).strip()
    else:
        # No fences found, return original text (it might be raw JSON)
        return txt.strip()

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _get_area_type_from_camera(camera_id: str) -> str:
    """Extract area type from camera ID for appropriate prompting"""
    camera_lower = camera_id.lower()
    if "general_labor" in camera_lower or "labor" in camera_lower:
        return "general_labor"
    elif "sandblast" in camera_lower:
        return "sandblast" 
    elif "powder" in camera_lower or "booth" in camera_lower:
        return "powder_booth"
    else:
        return "general_labor"  # Default

def _get_area_specific_prompt(area_type: str) -> str:
    """Generate area-specific prompts with emphasis on movement detection across 4 frames"""
    
    if area_type == "general_labor":
        return """You are analyzing a GENERAL LABOR area by examining a 4-FRAME SEQUENCE to detect MOVEMENT and ACTIVITY.

🎯 ANALYZE THE 4-FRAME SEQUENCE FOR MOVEMENT:
Frame 1 → Frame 2 → Frame 3 → Frame 4

LOOK FOR THESE ACTIVITIES (only if movement is detected):
- Manual hanging: Workers actually MOVING to lift/place parts on racks
- Forklift hanging: Forklifts actually MOVING to position parts on racks  
- Manual removing: Workers actually MOVING to take parts off racks
- Forklift removing: Forklifts actually MOVING to remove parts from racks
- Rack maintenance: Workers actively working on racks

⚠️ IMPORTANT MOVEMENT RULES:
- If a forklift is STATIONARY (not moving between frames) → report "other" with explanation "stationary equipment"
- If workers are STATIONARY (not actively working) → report "other" with explanation "no activity detected"
- If equipment is present but NOT NEAR RACKS → report "other" with explanation "equipment not engaged with racks"
- Only report hanging/removing if you see ACTUAL MOVEMENT toward/away from racks

RESPOND WITH JSON:
{
  "events": [
    {
      "operation_type": "manual_hanging|forklift_hanging|manual_removing|forklift_removing|other",
      "explanation": "Describe what MOVEMENT or ACTIVITY you observed across the 4 frames. If no movement, state 'no movement detected' or 'stationary equipment'",
      "confidence": 0.85,
      "forklift_detected": true/false,
      "manual_detected": true/false,
      "people_count": 0,
      "action": "hanging parts|removing parts|maintenance|no activity|other",
      "rack_id": "rack_01|N/A",
      "zone": "left|center|right|general"
    }
  ]
}

FOCUS: Only report activities where you see ACTUAL MOVEMENT between frames. Static scenes = "no activity"."""

    elif area_type == "powder_booth":
        return """You are analyzing a POWDER COATING BOOTH by examining a 4-FRAME SEQUENCE to detect MOVEMENT and ACTIVITY.

🎯 ANALYZE THE 4-FRAME SEQUENCE FOR MOVEMENT:
Frame 1 → Frame 2 → Frame 3 → Frame 4

LOOK FOR THESE ACTIVITIES (only if movement is detected):
- Workers ENTERING the booth (movement from outside to inside)
- Workers EXITING the booth (movement from inside to outside)
- Active painting/coating operations (spray gun movement, worker motion)
- Parts being MOVED for preparation
- Equipment being actively operated

⚠️ IMPORTANT MOVEMENT RULES:
- If the booth appears STATIC (no movement between frames) → report "other" with explanation "no activity detected"
- If a worker is STATIONARY inside → report "other" with explanation "worker stationary in booth"
- Only report specific activities if you see ACTUAL MOVEMENT or ACTIVE WORK

⚠️ IMPORTANT: This is NOT a hanging area. Parts hanging happens in General Labor BEFORE coming here.

RESPOND WITH JSON:
{
  "events": [
    {
      "operation_type": "other",
      "explanation": "Describe what MOVEMENT or ACTIVITY you observed across the 4 frames. If no movement, state 'no movement detected'",
      "confidence": 0.85,
      "forklift_detected": false,
      "manual_detected": true/false,
      "people_count": 0,
      "action": "painting|coating|booth_entry|booth_exit|preparation|no activity|other",
      "rack_id": "N/A",
      "zone": "booth_interior|booth_entrance|preparation_area"
    }
  ]
}

FOCUS: Only report activities where you see ACTUAL MOVEMENT. Static scenes = "no activity"."""

    elif area_type == "sandblast":
        return """You are analyzing a SANDBLAST area by examining a 4-FRAME SEQUENCE to detect MOVEMENT and ACTIVITY.

🎯 ANALYZE THE 4-FRAME SEQUENCE FOR MOVEMENT:
Frame 1 → Frame 2 → Frame 3 → Frame 4

LOOK FOR THESE ACTIVITIES (only if movement is detected):
- Workers ACTIVELY handling individual parts (movement of parts)
- Sandblasting operations IN PROGRESS (visible spray, movement)
- Parts being MOVED for preparation or inspection
- Workers MOVING between work areas
- Equipment being actively operated

⚠️ IMPORTANT MOVEMENT RULES:
- If the area appears STATIC (no movement between frames) → report "other" with explanation "no activity detected"
- If equipment is visible but NOT OPERATING → report "other" with explanation "equipment idle"
- If a worker is STATIONARY → report "other" with explanation "worker stationary"
- Only report specific activities if you see ACTUAL MOVEMENT or ACTIVE WORK

⚠️ IMPORTANT: This is NOT a hanging area. This area processes individual parts, not rack hanging.

RESPOND WITH JSON:
{
  "events": [
    {
      "operation_type": "other",
      "explanation": "Describe what MOVEMENT or ACTIVITY you observed across the 4 frames. If no movement, state 'no movement detected'",
      "confidence": 0.85,
      "forklift_detected": false,
      "manual_detected": true/false,
      "people_count": 0,
      "action": "sandblasting|part_preparation|inspection|maintenance|no activity|other",
      "rack_id": "N/A",
      "zone": "blast_booth|preparation_area|inspection_area"
    }
  ]
}

FOCUS: Only report activities where you see ACTUAL MOVEMENT. Static scenes = "no activity"."""

    else:
        # Fallback for unknown areas
        return """You are analyzing an industrial work area by examining a 4-FRAME SEQUENCE to detect MOVEMENT and ACTIVITY.

🎯 ANALYZE THE 4-FRAME SEQUENCE FOR MOVEMENT:
Frame 1 → Frame 2 → Frame 3 → Frame 4

Only report activities if you observe ACTUAL MOVEMENT between frames.

RESPOND WITH JSON:
{
  "events": [
    {
      "operation_type": "other",
      "explanation": "Describe what MOVEMENT or ACTIVITY you observed across the 4 frames. If no movement, state 'no movement detected'",
      "confidence": 0.70,
      "forklift_detected": true/false,
      "manual_detected": true/false,
      "people_count": 0,
      "action": "work_activity|no activity",
      "rack_id": "N/A",
      "zone": "unknown"
    }
  ]
}

FOCUS: Only report activities where you see ACTUAL MOVEMENT. Static scenes = "no activity"."""

def _parse_enhanced_response(response_text: str, area_type: str) -> List[Dict]:
    """
    FIXED: Parse the enhanced JSON response with robust error handling.
    
    This function now properly handles both raw JSON and markdown-fenced JSON responses.
    """
    
    # Log the parsing attempt
    log_parsing_attempt("action_detection", response_text, None, "initial_json")
    llm_logger.info(f"=== PARSING RESPONSE FOR {area_type} ===")
    llm_logger.info(f"Raw response length: {len(response_text)}")
    llm_logger.info(f"Raw response preview: {response_text[:300]}...")
    
    def fix_incomplete_json(text):
        llm_logger.debug("Attempting to fix incomplete JSON")
        text = text.strip()
        
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
    
    # STEP 1: Try to parse the response as raw JSON (most common case)
    try:
        llm_logger.info("ATTEMPT 1: Parsing raw response as JSON")
        data = json.loads(response_text)
        
        # Handle both formats: {"events": [...]} and [...]
        if isinstance(data, list):
            events = data  # Direct list format
            llm_logger.info("✅ SUCCESS: Raw response is direct list format")
        elif isinstance(data, dict) and "events" in data:
            events = data["events"]  # Wrapped format
            llm_logger.info("✅ SUCCESS: Raw response is wrapped format")
        else:
            llm_logger.warning("⚠️ Raw response is unexpected JSON format")
            events = []
            
        log_parsing_attempt("action_detection", response_text, events, "successful_raw_json")
        
    except json.JSONDecodeError as e:
        llm_logger.info(f"ATTEMPT 1 FAILED: {e}")
        
        # STEP 2: Try to strip markdown fences and parse
        try:
            llm_logger.info("ATTEMPT 2: Stripping fences and parsing")
            clean_response = _strip_fence(response_text)
            llm_logger.info(f"Cleaned response: {clean_response[:300]}...")
            
            data = json.loads(clean_response)
            
            # Handle both formats for cleaned JSON too
            if isinstance(data, list):
                events = data
                llm_logger.info("✅ SUCCESS: Cleaned response is direct list format")
            elif isinstance(data, dict) and "events" in data:
                events = data["events"]
                llm_logger.info("✅ SUCCESS: Cleaned response is wrapped format")
            else:
                events = []
                llm_logger.warning("⚠️ Cleaned response is unexpected format")
                
            log_parsing_attempt("action_detection", clean_response, events, "successful_fence_stripped")
            
        except json.JSONDecodeError as e2:
            llm_logger.info(f"ATTEMPT 2 FAILED: {e2}")
            
            # STEP 3: Try to fix common JSON issues
            try:
                llm_logger.info("ATTEMPT 3: Fixing JSON and parsing")
                fixed_json = fix_incomplete_json(clean_response if 'clean_response' in locals() else response_text)
                llm_logger.info(f"Fixed JSON: {fixed_json[:300]}...")
                
                data = json.loads(fixed_json)
                
                if isinstance(data, list):
                    events = data
                elif isinstance(data, dict) and "events" in data:
                    events = data["events"]
                else:
                    events = []
                    
                log_parsing_attempt("action_detection", fixed_json, events, "successful_fixed_json")
                llm_logger.info("✅ SUCCESS: Fixed JSON parsed successfully")
                
            except json.JSONDecodeError as e3:
                llm_logger.error(f"ATTEMPT 3 FAILED: {e3}")
                log_parsing_attempt("action_detection", fixed_json if 'fixed_json' in locals() else response_text, None, "failed_all_json")
                
                # STEP 4: Last resort - extract partial information
                llm_logger.info("ATTEMPT 4: Partial extraction")
                events = _extract_partial_response(response_text)
                log_parsing_attempt("action_detection", response_text, events, "partial_extraction")
    
    # Validate events are appropriate for this area and show actual movement
    validated_events = _validate_area_response(events, area_type)
    
    # Process validated events
    normalized_events = []
    ts = _now_iso()
    
    for i, event in enumerate(validated_events):
        llm_logger.debug(f"Processing event {i}: {event}")
        
        # Validate required fields and provide defaults
        operation_type = event.get("operation_type", "other")
        if operation_type not in ["manual_hanging", "forklift_hanging", 
                                "manual_removing", "forklift_removing", "other"]:
            llm_logger.warning(f"Invalid operation_type '{operation_type}', defaulting to 'other'")
            operation_type = "other"
        
        # Check for movement-related explanations
        explanation = event.get("explanation", "No explanation provided")
        action = event.get("action", "unknown")
        
        # Detect if this is a "no activity" response
        no_activity_indicators = [
            "no movement detected", "stationary equipment", "no activity detected",
            "worker stationary", "equipment idle", "no activity", "static scene"
        ]
        
        is_no_activity = any(indicator in explanation.lower() for indicator in no_activity_indicators)
        is_no_activity = is_no_activity or action.lower() in ["no activity", "no movement"]
        
        if is_no_activity:
            llm_logger.info(f"Detected 'no activity' response: {explanation}")
            # Ensure these are marked appropriately
            operation_type = "other"
            if action.lower() not in ["no activity", "no movement"]:
                action = "no activity"
        
        normalized_event = {
            "timestamp": ts,
            "camera_id": "",  # Will be filled by caller
            "rack_id": event.get("rack_id", "N/A"),
            "people_count": max(0, int(event.get("people_count", 0))),
            "action": action,
            "confidence": min(1.0, max(0.0, float(event.get("confidence", 0.0)))),
            "zone": event.get("zone", "unknown"),
            
            # Enhanced fields
            "operation_type": operation_type,
            "explanation": explanation,
            "forklift_detected": 1 if event.get("forklift_detected", False) else 0,
            "manual_detected": 1 if event.get("manual_detected", False) else 0,
        }
        
        normalized_events.append(normalized_event)
        llm_logger.debug(f"Normalized event {i}: {normalized_event}")
    
    llm_logger.info(f"✅ PARSING COMPLETE: {len(normalized_events)} events for {area_type} area")
    return normalized_events

def _validate_area_response(events: List[Dict], area_type: str) -> List[Dict]:
    """Validate that detected events are appropriate for the area type and show actual movement"""
    
    validated_events = []
    
    for event in events:
        operation_type = event.get("operation_type", "other")
        explanation = event.get("explanation", "")
        action = event.get("action", "unknown")
        
        # Check if operation type is valid for this area
        if area_type == "general_labor":
            # Only general labor should detect hanging/removing operations
            if operation_type in ["manual_hanging", "forklift_hanging", "manual_removing", "forklift_removing", "other"]:
                # Additional validation: check if explanation suggests actual movement
                movement_keywords = ["moving", "lifting", "positioning", "placing", "carrying", "transporting"]
                static_keywords = ["stationary", "idle", "not moving", "static", "no movement", "no activity"]
                
                has_movement = any(keyword in explanation.lower() for keyword in movement_keywords)
                is_static = any(keyword in explanation.lower() for keyword in static_keywords)
                
                if operation_type in ["manual_hanging", "forklift_hanging", "manual_removing", "forklift_removing"] and is_static:
                    llm_logger.warning(f"Detected {operation_type} but explanation suggests no movement: '{explanation}'")
                    event["operation_type"] = "other"
                    event["action"] = "no activity"
                    event["explanation"] = f"No active {operation_type.split('_')[1]} detected - {explanation}"
                
                validated_events.append(event)
            else:
                llm_logger.warning(f"Invalid operation_type '{operation_type}' for general_labor area, converting to 'other'")
                event["operation_type"] = "other"
                event["explanation"] = f"General labor activity: {explanation}"
                validated_events.append(event)
                
        elif area_type in ["powder_booth", "sandblast"]:
            # Powder booth and sandblast should NEVER detect hanging operations
            if operation_type in ["manual_hanging", "forklift_hanging", "manual_removing", "forklift_removing"]:
                llm_logger.warning(f"Detected hanging operation '{operation_type}' in {area_type} area - this is incorrect!")
                # Convert to appropriate area activity
                if area_type == "powder_booth":
                    event["operation_type"] = "other"
                    event["action"] = "powder_coating_activity"
                    event["explanation"] = f"Powder coating area activity (not hanging): {explanation}"
                else:  # sandblast
                    event["operation_type"] = "other"
                    event["action"] = "sandblast_activity" 
                    event["explanation"] = f"Sandblast area activity (not hanging): {explanation}"
                validated_events.append(event)
            else:
                # Valid operation for this area - but still check for movement
                static_keywords = ["stationary", "idle", "not moving", "static", "no movement", "no activity"]
                is_static = any(keyword in explanation.lower() for keyword in static_keywords)
                
                if is_static:
                    llm_logger.info(f"Detected no activity in {area_type}: {explanation}")
                    event["action"] = "no activity"
                
                validated_events.append(event)
        else:
            # Unknown area, allow anything but log it
            llm_logger.info(f"Unknown area type '{area_type}', allowing operation '{operation_type}'")
            validated_events.append(event)
    
    return validated_events

def _extract_partial_response(text: str) -> List[Dict]:
    """Extract information even from malformed JSON response."""
    llm_logger.info("Attempting partial extraction from malformed response")
    
    # Extract operation type
    operation_type = "other"
    if "manual_hanging" in text:
        operation_type = "manual_hanging"
    elif "forklift_hanging" in text:
        operation_type = "forklift_hanging"
    elif "manual_removing" in text:
        operation_type = "manual_removing"
    elif "forklift_removing" in text:
        operation_type = "forklift_removing"
    
    llm_logger.debug(f"Extracted operation_type: {operation_type}")
    
    # Extract explanation (look for explanation field)
    explanation_match = re.search(r'"explanation":\s*"([^"]*)"', text)
    explanation = explanation_match.group(1) if explanation_match else "Partial extraction from incomplete response"
    llm_logger.debug(f"Extracted explanation: {explanation}")
    
    # Extract confidence
    confidence_match = re.search(r'"confidence":\s*([0-9.]+)', text)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.7
    llm_logger.debug(f"Extracted confidence: {confidence}")
    
    # Extract boolean fields
    forklift_detected = "forklift_detected.*true" in text.lower() or "forklift" in text.lower()
    manual_detected = "manual_detected.*true" in text.lower() or "person" in text.lower() or "worker" in text.lower()
    
    llm_logger.debug(f"Extracted forklift_detected: {forklift_detected}")
    llm_logger.debug(f"Extracted manual_detected: {manual_detected}")
    
    # Extract people count
    people_match = re.search(r'"people_count":\s*([0-9]+)', text)
    people_count = int(people_match.group(1)) if people_match else (1 if manual_detected else 0)
    llm_logger.debug(f"Extracted people_count: {people_count}")
    
    result = [{
        "timestamp": _now_iso(),
        "camera_id": "",
        "rack_id": "N/A",
        "people_count": people_count,
        "action": "hanging parts" if "hanging" in operation_type else "removing parts" if "removing" in operation_type else "work activity",
        "confidence": confidence,
        "zone": "area",
        "operation_type": operation_type,
        "explanation": explanation,
        "forklift_detected": 1 if forklift_detected else 0,
        "manual_detected": 1 if manual_detected else 0,
    }]
    
    llm_logger.info(f"Partial extraction successful: {result}")
    return result

def detect_rack_actions(image_paths: List[str],
                        camera_id:   str,
                        timeout:     int = 120,
                        use_examples: bool = True) -> List[Dict]:
    """
    Enhanced rack action detection with area-specific prompts and validation
    """
    
    llm_logger.info(f"=== STARTING AREA-SPECIFIC ACTION DETECTION ===")
    llm_logger.info(f"Camera: {camera_id}")
    llm_logger.info(f"Images: {len(image_paths)}")
    llm_logger.info(f"Timeout: {timeout}s")
    llm_logger.info(f"Use examples: {use_examples}")
    
    # Log image paths and their existence
    for i, path in enumerate(image_paths):
        exists = Path(path).exists()
        size = Path(path).stat().st_size if exists else 0
        llm_logger.info(f"Image {i+1}: {path} (exists: {exists}, size: {size} bytes)")
    
    logger.info("Enhanced action detection: analyzing %d frames for %s", len(image_paths), camera_id)
    
    # Get area-specific configuration
    area_type = _get_area_type_from_camera(camera_id)
    llm_logger.info(f"Detected area type: {area_type}")
    
    # Use area-specific prompt instead of generic examples
    prompt = _get_area_specific_prompt(area_type)
    llm_logger.info(f"Using area-specific prompt for {area_type}")
    
    # Log the prompt being used (truncated for readability)
    prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
    llm_logger.info(f"Prompt preview: {prompt_preview}")
    
    # Build content with images
    content = [{"type": "text", "text": prompt}]
    valid_images = 0
    
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
            log_error_details("action_detection", e, {"image_path": str(p)})

    if len(content) == 1:  # Only text, no valid images
        llm_logger.error("No valid images found for analysis")
        logger.error("No valid images found for analysis")
        return []

    llm_logger.info(f"Successfully prepared {valid_images} images for analysis")

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 1200,
    }

    # Log the full request AND save the images
    log_llm_request("action_detection", payload, valid_images, camera_id)

    try:
        llm_logger.info(f"Sending request to LM-Studio: {LM_STUDIO_URL}/chat/completions")
        logger.debug("Sending request to LM-Studio with %d images", len(content) - 1)
        
        start_time = time.time()
        r = requests.post(f"{LM_STUDIO_URL}/chat/completions",
                          json=payload, timeout=timeout)
        processing_time = time.time() - start_time
        
        llm_logger.info(f"Request completed in {processing_time:.2f}s with status {r.status_code}")
        
        r.raise_for_status()
    except Exception as exc:
        log_error_details("action_detection", exc, {
            "url": f"{LM_STUDIO_URL}/chat/completions",
            "timeout": timeout,
            "image_count": valid_images
        })
        logger.exception("Enhanced action detection: HTTP failure – %s", exc)
        return []

    try:
        body = r.json()
        log_llm_response("action_detection", body, body.get("choices", [{}])[0].get("message", {}).get("content", ""), processing_time)
    except ValueError as e:
        llm_logger.error(f"Failed to parse response as JSON: {e}")
        llm_logger.error(f"Raw response text: {r.text[:1000]}")
        logger.error("Enhanced action detection: plain-text LM response: %.150s …",
                     r.text.replace('\\n', ' '))
        return []

    if "choices" not in body:
        llm_logger.error(f"No 'choices' in response body: {json.dumps(body)[:400]}")
        logger.error("Enhanced action detection: LM error – %s", json.dumps(body)[:400])
        return []

    raw_response = body["choices"][0]["message"]["content"]
    
    llm_logger.info(f"Raw LM response: {raw_response}")
    logger.debug("Enhanced action detection: raw LM content for %s: %.300s", camera_id, raw_response)

    # Parse with enhanced parser that includes area validation and FIXED JSON parsing
    events = _parse_enhanced_response(raw_response, area_type)
    
    # Fill in camera_id for all events
    for event in events:
        event["camera_id"] = camera_id
    
    if events:
        llm_logger.info(f"Successfully detected {len(events)} events in {camera_id} ({area_type} area)")
        logger.info("Enhanced action detection: detected %d events in %s", len(events), camera_id)
        for event in events:
            llm_logger.info(f"Event: {event['operation_type']}: {event['action']} (confidence: {event['confidence']:.2f}, forklift: {bool(event['forklift_detected'])}, manual: {bool(event['manual_detected'])})")
            logger.info("  - %s: %s (confidence: %.2f, forklift: %s, manual: %s)", 
                       event["operation_type"], event["action"], 
                       event["confidence"], bool(event["forklift_detected"]), 
                       bool(event["manual_detected"]))
    else:
        llm_logger.info(f"No events detected in {camera_id} ({area_type} area)")
        logger.debug("Enhanced action detection: no events detected in %s", camera_id)
    
    llm_logger.info(f"=== AREA-SPECIFIC ACTION DETECTION COMPLETE ({area_type.upper()}) ===")
    return events