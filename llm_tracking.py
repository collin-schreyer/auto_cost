"""
Enhanced llm_tracking.py with image saving and viewing capabilities
"""

import logging
import json
import os
import shutil
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Create a dedicated logger for LLM interactions
llm_logger = logging.getLogger("llm_interactions")

def setup_llm_tracking():
    """Set up comprehensive LLM request/response tracking with image saving"""
    
    # Create logs and images directories
    logs_dir = Path("logs")
    images_dir = Path("logs/images")
    logs_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # Configure file handler for LLM interactions
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = logs_dir / f"llm_interactions_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    llm_logger.addHandler(file_handler)
    llm_logger.setLevel(logging.DEBUG)
    
    # Also log to console if debug mode
    if os.getenv("DEBUG_LLM", "false").lower() == "true":
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        llm_logger.addHandler(console_handler)
    
    return llm_logger

def save_images_from_request(module_name: str, payload: dict, camera_id: str = "") -> List[str]:
    """
    Extract and save images from LLM request payload
    Returns list of saved image paths for viewing
    """
    
    saved_image_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    images_dir = Path("logs/images")
    
    try:
        messages = payload.get("messages", [])
        
        for msg_idx, message in enumerate(messages):
            if message.get("role") == "user":
                content = message.get("content", [])
                
                if isinstance(content, list):
                    image_count = 0
                    
                    for item in content:
                        if item.get("type") == "image_url":
                            image_url = item.get("image_url", {}).get("url", "")
                            
                            if image_url.startswith("data:image/"):
                                try:
                                    # Extract base64 data
                                    header, base64_data = image_url.split(",", 1)
                                    
                                    # Determine file extension from header
                                    if "png" in header:
                                        ext = "png"
                                    elif "jpg" in header or "jpeg" in header:
                                        ext = "jpg"
                                    else:
                                        ext = "png"  # default
                                    
                                    # Create filename
                                    camera_part = f"{camera_id}_" if camera_id else ""
                                    filename = f"{module_name}_{camera_part}{timestamp}_frame_{image_count + 1}.{ext}"
                                    file_path = images_dir / filename
                                    
                                    # Decode and save image
                                    image_data = base64.b64decode(base64_data)
                                    with open(file_path, "wb") as f:
                                        f.write(image_data)
                                    
                                    saved_image_paths.append(str(file_path))
                                    image_count += 1
                                    
                                    llm_logger.debug(f"Saved image: {filename} ({len(image_data)} bytes)")
                                    
                                except Exception as e:
                                    llm_logger.error(f"Failed to save image {image_count}: {e}")
                                    continue
        
        if saved_image_paths:
            llm_logger.info(f"Saved {len(saved_image_paths)} images for {module_name} request")
        
    except Exception as e:
        llm_logger.error(f"Error saving images from request: {e}")
    
    return saved_image_paths

def log_llm_request(module_name: str, payload: dict, image_count: int = 0, camera_id: str = ""):
    """Log the full request being sent to LM Studio and save images"""
    
    # Save images from the request
    saved_images = save_images_from_request(module_name, payload, camera_id)
    
    # Create a safe copy of payload (without base64 images)
    safe_payload = payload.copy()
    
    if "messages" in safe_payload:
        safe_messages = []
        for msg in safe_payload["messages"]:
            safe_msg = {"role": msg["role"]}
            
            if isinstance(msg["content"], list):
                safe_content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        safe_content.append(item)
                    elif item["type"] == "image_url":
                        # Replace base64 with placeholder and reference to saved file
                        safe_content.append({
                            "type": "image_url",
                            "image_url": {"url": "[BASE64_IMAGE_DATA_SAVED]"},
                            "saved_images": saved_images  # Add reference to saved images
                        })
                safe_msg["content"] = safe_content
            else:
                safe_msg["content"] = msg["content"]
            
            safe_messages.append(safe_msg)
        
        safe_payload["messages"] = safe_messages
    
    llm_logger.info(f"=== {module_name.upper()} REQUEST ===")
    llm_logger.info(f"Camera: {camera_id}")
    llm_logger.info(f"Model: {payload.get('model', 'unknown')}")
    llm_logger.info(f"Temperature: {payload.get('temperature', 'unknown')}")
    llm_logger.info(f"Max tokens: {payload.get('max_tokens', 'unknown')}")
    llm_logger.info(f"Images attached: {image_count}")
    llm_logger.info(f"Saved images: {saved_images}")
    llm_logger.info(f"Full payload: {json.dumps(safe_payload, indent=2)}")

def log_llm_response(module_name: str, response_data: dict, raw_text: str, 
                    processing_time: float = None):
    """Log the full response from LM Studio"""
    
    llm_logger.info(f"=== {module_name.upper()} RESPONSE ===")
    if processing_time:
        llm_logger.info(f"Processing time: {processing_time:.2f}s")
    
    # Log usage statistics if available
    usage = response_data.get("usage", {})
    if usage:
        llm_logger.info(f"Token usage: {usage}")
    
    # Log the raw response
    llm_logger.info(f"Raw response text: {raw_text}")
    
    # Log the full response structure
    llm_logger.info(f"Full response: {json.dumps(response_data, indent=2)}")

def log_parsing_attempt(module_name: str, raw_text: str, parsed_result: any, 
                       parsing_method: str = "json"):
    """Log parsing attempts and results"""
    
    llm_logger.info(f"=== {module_name.upper()} PARSING ===")
    llm_logger.info(f"Parsing method: {parsing_method}")
    llm_logger.info(f"Raw text to parse: {raw_text}")
    llm_logger.info(f"Parsed result: {parsed_result}")
    llm_logger.info(f"Parsing successful: {parsed_result is not None}")

def log_error_details(module_name: str, error: Exception, context: dict = None):
    """Log detailed error information"""
    
    llm_logger.error(f"=== {module_name.upper()} ERROR ===")
    llm_logger.error(f"Error type: {type(error).__name__}")
    llm_logger.error(f"Error message: {str(error)}")
    
    if context:
        llm_logger.error(f"Error context: {json.dumps(context, indent=2)}")
    
    llm_logger.exception("Full stack trace:")

def get_saved_images_for_timestamp(module_name: str, timestamp_str: str, camera_id: str = "") -> List[str]:
    """
    Get list of saved image files for a specific timestamp
    """
    images_dir = Path("logs/images")
    if not images_dir.exists():
        return []
    
    # Convert timestamp for filename matching
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', ''))
        time_pattern = dt.strftime("%Y%m%d_%H%M%S")
    except:
        # Fallback to searching by partial timestamp
        time_pattern = timestamp_str.replace(':', '').replace('-', '')[:13]
    
    # Search for matching images
    camera_part = f"{camera_id}_" if camera_id else ""
    pattern = f"{module_name}_{camera_part}{time_pattern}*"
    
    matching_images = list(images_dir.glob(pattern))
    return sorted([str(img) for img in matching_images])

def cleanup_old_images(days_to_keep: int = 7):
    """
    Clean up old saved images to prevent disk space issues
    """
    images_dir = Path("logs/images")
    if not images_dir.exists():
        return
    
    cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
    deleted_count = 0
    
    try:
        for image_file in images_dir.glob("*.png"):
            if image_file.stat().st_mtime < cutoff_time:
                image_file.unlink()
                deleted_count += 1
        
        for image_file in images_dir.glob("*.jpg"):
            if image_file.stat().st_mtime < cutoff_time:
                image_file.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            llm_logger.info(f"Cleaned up {deleted_count} old image files")
            
    except Exception as e:
        llm_logger.error(f"Error during image cleanup: {e}")

# Initialize the logger
setup_llm_tracking()

# Export the main functions
__all__ = ['llm_logger', 'log_llm_request', 'log_llm_response', 
           'log_parsing_attempt', 'log_error_details', 'get_saved_images_for_timestamp',
           'cleanup_old_images']