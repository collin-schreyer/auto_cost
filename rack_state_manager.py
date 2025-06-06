"""
rack_state_manager.py - Core logic for global rack state management
-----------------------------------------------------------------
Manages the position-based tracking of racks across general_labor and powder_booth zones.
Handles state persistence, rack identity assignment, and provides context for detection.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from database import (
    get_global_rack_state,
    update_global_rack_position,
    get_racks_in_camera,
    mark_rack_missing,
    add_rack_movement,
    get_recent_movements,
    get_zone_definitions,
    add_zone_definition,
    _now_iso
)

logger = logging.getLogger(__name__)

# Zone configuration for position-based tracking
ZONE_CONFIGS = {
    "general_labor": {
        "zones": ["left_side", "center_left", "center_right", "right_side"],
        "transition_zones": ["right_side"],  # Where racks exit to powder_booth
        "entrance_zones": ["left_side"]      # Where racks enter from powder_booth
    },
    "powder_booth": {
        "zones": ["entrance", "center", "back_wall"],
        "transition_zones": ["entrance"],    # Where racks exit to general_labor
        "entrance_zones": ["entrance"]       # Where racks enter from general_labor
    }
}

class RackStateManager:
    """
    Core manager for global rack state across zones.
    """
    
    def __init__(self):
        self._initialize_zones()
    
    def _initialize_zones(self):
        """Initialize zone definitions in database if not already present"""
        try:
            for camera_id, config in ZONE_CONFIGS.items():
                existing_zones = get_zone_definitions(camera_id)
                existing_zone_names = {z['zone_name'] for z in existing_zones}
                
                for zone_name in config['zones']:
                    if zone_name not in existing_zone_names:
                        is_transition = zone_name in config.get('transition_zones', [])
                        description = f"{zone_name.replace('_', ' ').title()} area of {camera_id}"
                        
                        add_zone_definition(
                            camera_id=camera_id,
                            zone_name=zone_name,
                            description=description,
                            is_transition_zone=is_transition
                        )
                        logger.info(f"Initialized zone: {camera_id}/{zone_name}")
        except Exception as e:
            logger.error(f"Failed to initialize zones: {e}")
    
    def get_current_rack_context(self, camera_id: str) -> Dict:
        """
        Get context about racks currently in this camera zone for prompt building.
        
        Returns:
            Dict with current racks, recent movements, and zone info
        """
        try:
            # Get current racks in this camera
            current_racks = get_racks_in_camera(camera_id)
            
            # Get recent movements that might affect this camera
            recent_movements = get_recent_movements(hours=4)
            
            # Filter movements relevant to this camera
            relevant_movements = [
                m for m in recent_movements 
                if m['to_camera'] == camera_id or m['from_camera'] == camera_id
            ]
            
            # Get racks that recently left this camera (might return)
            recently_departed = [
                m for m in relevant_movements 
                if m['from_camera'] == camera_id
            ]
            
            # Get racks that recently arrived
            recently_arrived = [
                m for m in relevant_movements 
                if m['to_camera'] == camera_id
            ]
            
            return {
                "camera_id": camera_id,
                "current_racks": current_racks,
                "recently_departed": recently_departed,
                "recently_arrived": recently_arrived,
                "total_active_racks": len(current_racks),
                "zone_config": ZONE_CONFIGS.get(camera_id, {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get rack context for {camera_id}: {e}")
            return {
                "camera_id": camera_id,
                "current_racks": [],
                "recently_departed": [],
                "recently_arrived": [],
                "total_active_racks": 0,
                "zone_config": {}
            }
    
    def assign_rack_ids(self, detected_racks: List[Dict], camera_id: str) -> List[Dict]:
        """
        Assign global rack IDs to newly detected racks based on position and context.
        
        Args:
            detected_racks: List of racks detected by vision model with positions
            camera_id: Camera where racks were detected
            
        Returns:
            List of racks with assigned global_rack_id fields
        """
        try:
            context = self.get_current_rack_context(camera_id)
            current_racks = {r['rack_id']: r for r in context['current_racks']}
            
            assigned_racks = []
            
            for detected_rack in detected_racks:
                local_id = detected_rack.get('rack_id', 'unknown')
                zone = detected_rack.get('zone_description', 'unknown')
                
                # Try to match with existing racks in same zone
                global_rack_id = self._match_to_existing_rack(
                    detected_rack, current_racks, camera_id
                )
                
                if not global_rack_id:
                    # Check if this might be a rack that recently moved here
                    global_rack_id = self._check_for_transferred_rack(
                        detected_rack, context['recently_arrived'], camera_id
                    )
                
                if not global_rack_id:
                    # Create new global rack ID
                    global_rack_id = self._create_new_global_rack_id()
                    logger.info(f"Created new global rack: {global_rack_id} in {camera_id}/{zone}")
                
                # Update the global rack state
                self._update_rack_state(global_rack_id, detected_rack, camera_id)
                
                # Add global ID to the detected rack
                detected_rack['global_rack_id'] = global_rack_id
                assigned_racks.append(detected_rack)
                
                logger.debug(f"Assigned {local_id} → {global_rack_id} in {camera_id}/{zone}")
            
            # Check for racks that disappeared
            self._check_for_missing_racks(assigned_racks, current_racks, camera_id)
            
            return assigned_racks
            
        except Exception as e:
            logger.error(f"Failed to assign rack IDs for {camera_id}: {e}")
            # Return racks with temporary IDs
            return [
                {**rack, 'global_rack_id': f"temp_{camera_id}_{i}"} 
                for i, rack in enumerate(detected_racks)
            ]
    
    def _match_to_existing_rack(self, detected_rack: Dict, current_racks: Dict, camera_id: str) -> Optional[str]:
        """
        Try to match detected rack to existing rack in same camera based on position.
        """
        detected_zone = detected_rack.get('zone_description', '')
        
        # Look for racks in same zone
        for global_id, existing_rack in current_racks.items():
            if existing_rack['current_zone'] == detected_zone:
                logger.debug(f"Matched rack in {detected_zone} to existing {global_id}")
                return existing_rack['rack_id']
        
        return None
    
    def _check_for_transferred_rack(self, detected_rack: Dict, recent_arrivals: List[Dict], camera_id: str) -> Optional[str]:
        """
        Check if this detected rack matches a rack that recently transferred to this camera.
        """
        detected_zone = detected_rack.get('zone_description', '')
        
        # Check arrivals in the last 2 hours
        recent_cutoff = datetime.now() - timedelta(hours=2)
        
        for movement in recent_arrivals:
            movement_time = datetime.fromisoformat(movement['movement_timestamp'].replace('Z', ''))
            
            if movement_time > recent_cutoff and movement['to_zone'] == detected_zone:
                logger.info(f"Matched transferred rack {movement['global_rack_id']} in {detected_zone}")
                return movement['global_rack_id']
        
        return None
    
    def _create_new_global_rack_id(self) -> str:
        """
        Generate a new unique global rack ID.
        """
        # Get existing rack IDs to find next available number
        all_racks = get_global_rack_state()
        existing_numbers = []
        
        for rack in all_racks:
            rack_id = rack['rack_id']
            if rack_id.startswith('global_rack_'):
                try:
                    num = int(rack_id.split('_')[-1])
                    existing_numbers.append(num)
                except ValueError:
                    continue
        
        # Find next available number
        next_num = 1
        while next_num in existing_numbers:
            next_num += 1
        
        return f"global_rack_{next_num:02d}"
    
    def _update_rack_state(self, global_rack_id: str, detected_rack: Dict, camera_id: str):
        """
        Update the global state for this rack.
        """
        zone = detected_rack.get('zone_description', 'unknown')
        position_desc = f"{zone} in {camera_id}"
        
        # Extract any visual notes if provided
        visual_notes = detected_rack.get('visual_notes', '')
        
        update_global_rack_position(
            rack_id=global_rack_id,
            camera_id=camera_id,
            zone=zone,
            position_description=position_desc,
            visual_notes=visual_notes
        )
    
    def _check_for_missing_racks(self, detected_racks: List[Dict], current_racks: Dict, camera_id: str):
        """
        Check if any previously tracked racks are now missing from detection.
        """
        detected_global_ids = {r.get('global_rack_id') for r in detected_racks}
        
        for existing_rack in current_racks.values():
            global_id = existing_rack['rack_id']
            
            if global_id not in detected_global_ids:
                # This rack is no longer detected
                # Check if it might have moved to transition zone
                zone_config = ZONE_CONFIGS.get(camera_id, {})
                current_zone = existing_rack['current_zone']
                
                if current_zone in zone_config.get('transition_zones', []):
                    # Rack was in transition zone, might have moved to other camera
                    logger.info(f"Rack {global_id} left transition zone {current_zone} in {camera_id}")
                    # Don't mark as missing yet, might appear in other camera
                else:
                    # Mark as potentially missing, but don't delete immediately
                    logger.warning(f"Rack {global_id} no longer detected in {camera_id}/{current_zone}")
    
    def build_prompt_context(self, camera_id: str) -> str:
        """
        Build the context section for the rack detection prompt.
        
        Returns:
            String with current rack state context for this camera
        """
        try:
            context = self.get_current_rack_context(camera_id)
            
            context_lines = [
                f"=== CURRENT RACK STATE FOR {camera_id.upper()} ==="
            ]
            
            # Current racks
            if context['current_racks']:
                context_lines.append("\nCurrently tracked racks:")
                for rack in context['current_racks']:
                    zone = rack['current_zone']
                    last_updated = rack['last_updated']
                    context_lines.append(f"• {rack['rack_id']}: Located in {zone} (last seen: {last_updated})")
            else:
                context_lines.append("\nNo racks currently tracked in this zone.")
            
            # Recent movements
            if context['recently_departed']:
                context_lines.append(f"\nRacks that recently left {camera_id}:")
                for movement in context['recently_departed'][-3:]:  # Last 3 movements
                    context_lines.append(f"• {movement['global_rack_id']}: Moved to {movement['to_camera']} at {movement['movement_timestamp']}")
            
            if context['recently_arrived']:
                context_lines.append(f"\nRacks that recently arrived in {camera_id}:")
                for movement in context['recently_arrived'][-3:]:  # Last 3 movements
                    context_lines.append(f"• {movement['global_rack_id']}: Came from {movement['from_camera']} at {movement['movement_timestamp']}")
            
            # Zone information
            zone_config = context['zone_config']
            if zone_config:
                zones = zone_config.get('zones', [])
                transition_zones = zone_config.get('transition_zones', [])
                context_lines.append(f"\nZone layout: {', '.join(zones)}")
                if transition_zones:
                    context_lines.append(f"Transition zones (racks may move to other cameras): {', '.join(transition_zones)}")
            
            return '\n'.join(context_lines)
            
        except Exception as e:
            logger.error(f"Failed to build prompt context for {camera_id}: {e}")
            return f"=== RACK CONTEXT ERROR FOR {camera_id.upper()} ===\nFailed to load rack state context."
    
    def validate_rack_assignments(self, camera_id: str) -> Dict:
        """
        Validate current rack assignments and detect potential issues.
        
        Returns:
            Dict with validation results and recommendations
        """
        try:
            context = self.get_current_rack_context(camera_id)
            current_racks = context['current_racks']
            
            issues = []
            recommendations = []
            
            # Check for racks in same zone
            zone_counts = {}
            for rack in current_racks:
                zone = rack['current_zone']
                zone_counts[zone] = zone_counts.get(zone, 0) + 1
            
            for zone, count in zone_counts.items():
                if count > 1:
                    issues.append(f"Multiple racks ({count}) detected in same zone: {zone}")
                    recommendations.append(f"Review rack positions in {zone} - may need better zone definition")
            
            # Check for very old rack states
            stale_cutoff = datetime.now() - timedelta(hours=8)
            for rack in current_racks:
                last_updated = datetime.fromisoformat(rack['last_updated'].replace('Z', ''))
                if last_updated < stale_cutoff:
                    issues.append(f"Rack {rack['rack_id']} hasn't been updated for {(datetime.now() - last_updated).total_seconds() / 3600:.1f} hours")
                    recommendations.append(f"Consider marking {rack['rack_id']} as missing if not physically present")
            
            # Check movement patterns
            recent_movements = get_recent_movements(hours=24)
            frequent_movers = {}
            for movement in recent_movements:
                rack_id = movement['global_rack_id']
                frequent_movers[rack_id] = frequent_movers.get(rack_id, 0) + 1
            
            for rack_id, move_count in frequent_movers.items():
                if move_count > 6:  # More than 6 movements in 24 hours
                    issues.append(f"Rack {rack_id} moved {move_count} times in 24 hours - unusually frequent")
                    recommendations.append(f"Check if {rack_id} is being tracked correctly or if there are detection issues")
            
            return {
                "camera_id": camera_id,
                "total_racks": len(current_racks),
                "total_issues": len(issues),
                "issues": issues,
                "recommendations": recommendations,
                "zone_distribution": zone_counts,
                "validation_timestamp": _now_iso()
            }
            
        except Exception as e:
            logger.error(f"Failed to validate rack assignments for {camera_id}: {e}")
            return {
                "camera_id": camera_id,
                "total_racks": 0,
                "total_issues": 1,
                "issues": [f"Validation failed: {str(e)}"],
                "recommendations": ["Check database connectivity and rack state manager"],
                "zone_distribution": {},
                "validation_timestamp": _now_iso()
            }
    
    def cleanup_stale_racks(self, hours_threshold: int = 12) -> int:
        """
        Mark racks as missing if they haven't been seen for a specified time.
        
        Args:
            hours_threshold: Hours after which to consider a rack missing
            
        Returns:
            Number of racks marked as missing
        """
        try:
            all_racks = get_global_rack_state()
            stale_cutoff = datetime.now() - timedelta(hours=hours_threshold)
            cleaned_count = 0
            
            for rack in all_racks:
                if rack['status'] != 'active':
                    continue
                    
                last_updated = datetime.fromisoformat(rack['last_updated'].replace('Z', ''))
                
                if last_updated < stale_cutoff:
                    mark_rack_missing(rack['rack_id'])
                    cleaned_count += 1
                    logger.info(f"Marked rack {rack['rack_id']} as missing (last seen: {last_updated})")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup stale racks: {e}")
            return 0


# Global instance for use by rack detection
rack_state_manager = RackStateManager()