"""
rack_transitions.py - Cross-zone movement detection and correlation
-----------------------------------------------------------------
Handles detection of racks moving between general_labor and powder_booth zones.
Manages the "rack appears in both cameras" scenario during transitions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from database import (
    get_global_rack_state,
    get_recent_movements,
    add_rack_movement,
    update_global_rack_position,
    _now_iso
)

logger = logging.getLogger(__name__)

# Movement correlation settings
CORRELATION_WINDOW_HOURS = 2  # How long to look for matching movements
TRANSITION_OVERLAP_MINUTES = 30  # How long racks can appear in both cameras
MIN_CORRELATION_CONFIDENCE = 0.7  # Minimum confidence for movement correlation

class RackTransitionManager:
    """
    Manages rack movements between zones and handles transition states.
    """
    
    def __init__(self):
        self.pending_transitions = {}  # Track racks potentially in transit
    
    def detect_cross_zone_movements(self, general_labor_racks: List[Dict], 
                                   powder_booth_racks: List[Dict]) -> List[Dict]:
        """
        Analyze rack detections from both cameras to identify cross-zone movements.
        
        Args:
            general_labor_racks: Racks detected in general_labor camera
            powder_booth_racks: Racks detected in powder_booth camera
            
        Returns:
            List of detected movement events
        """
        try:
            movements = []
            
            # Get current global state
            current_state = {r['rack_id']: r for r in get_global_rack_state()}
            
            # Check for racks that appear in unexpected zones
            gl_movements = self._check_unexpected_appearances(
                general_labor_racks, "general_labor", current_state
            )
            pb_movements = self._check_unexpected_appearances(
                powder_booth_racks, "powder_booth", current_state
            )
            
            movements.extend(gl_movements)
            movements.extend(pb_movements)
            
            # Check for racks that disappeared from expected zones
            disappearance_movements = self._check_disappearances(
                general_labor_racks, powder_booth_racks, current_state
            )
            movements.extend(disappearance_movements)
            
            # Handle racks appearing in both cameras (transition state)
            transition_movements = self._handle_dual_appearances(
                general_labor_racks, powder_booth_racks, current_state
            )
            movements.extend(transition_movements)
            
            # Log detected movements
            if movements:
                logger.info(f"Detected {len(movements)} potential rack movements")
                for movement in movements:
                    logger.info(f"  {movement['type']}: {movement.get('global_rack_id', 'unknown')} - {movement.get('description', 'no description')}")
            
            return movements
            
        except Exception as e:
            logger.error(f"Failed to detect cross-zone movements: {e}")
            return []
    
    def _check_unexpected_appearances(self, detected_racks: List[Dict], 
                                    camera_id: str, current_state: Dict) -> List[Dict]:
        """
        Check for racks that appear in a camera where they weren't expected.
        """
        movements = []
        
        for rack in detected_racks:
            global_rack_id = rack.get('global_rack_id')
            if not global_rack_id:
                continue
                
            current_rack = current_state.get(global_rack_id)
            if not current_rack:
                continue
                
            expected_camera = current_rack.get('current_camera')
            
            # If rack appears in different camera than expected
            if expected_camera and expected_camera != camera_id and expected_camera != 'in_transit':
                movements.append({
                    'type': 'unexpected_appearance',
                    'global_rack_id': global_rack_id,
                    'from_camera': expected_camera,
                    'to_camera': camera_id,
                    'from_zone': current_rack.get('current_zone', 'unknown'),
                    'to_zone': rack.get('zone_description', 'unknown'),
                    'confidence': 0.8,
                    'description': f"Rack appeared in {camera_id} but was expected in {expected_camera}",
                    'timestamp': _now_iso()
                })
                
                logger.info(f"Unexpected appearance: {global_rack_id} in {camera_id} (expected in {expected_camera})")
        
        return movements
    
    def _check_disappearances(self, general_labor_racks: List[Dict], 
                            powder_booth_racks: List[Dict], current_state: Dict) -> List[Dict]:
        """
        Check for racks that disappeared from their expected cameras.
        """
        movements = []
        
        # Get sets of global rack IDs currently detected
        gl_detected = {r.get('global_rack_id') for r in general_labor_racks if r.get('global_rack_id')}
        pb_detected = {r.get('global_rack_id') for r in powder_booth_racks if r.get('global_rack_id')}
        
        # Check each rack in current state
        for global_rack_id, rack_state in current_state.items():
            if rack_state.get('status') != 'active':
                continue
                
            current_camera = rack_state.get('current_camera')
            
            # Check if rack disappeared from its current camera
            if current_camera == 'general_labor' and global_rack_id not in gl_detected:
                # Check if it appeared in powder_booth
                if global_rack_id in pb_detected:
                    movements.append({
                        'type': 'cross_zone_movement',
                        'global_rack_id': global_rack_id,
                        'from_camera': 'general_labor',
                        'to_camera': 'powder_booth',
                        'from_zone': rack_state.get('current_zone', 'unknown'),
                        'to_zone': self._get_detected_zone(global_rack_id, powder_booth_racks),
                        'confidence': 0.9,
                        'description': f"Rack moved from general_labor to powder_booth",
                        'timestamp': _now_iso()
                    })
                else:
                    # Rack disappeared entirely
                    movements.append({
                        'type': 'disappearance',
                        'global_rack_id': global_rack_id,
                        'from_camera': 'general_labor',
                        'from_zone': rack_state.get('current_zone', 'unknown'),
                        'confidence': 0.7,
                        'description': f"Rack disappeared from general_labor",
                        'timestamp': _now_iso()
                    })
            
            elif current_camera == 'powder_booth' and global_rack_id not in pb_detected:
                # Check if it appeared in general_labor
                if global_rack_id in gl_detected:
                    movements.append({
                        'type': 'cross_zone_movement',
                        'global_rack_id': global_rack_id,
                        'from_camera': 'powder_booth',
                        'to_camera': 'general_labor',
                        'from_zone': rack_state.get('current_zone', 'unknown'),
                        'to_zone': self._get_detected_zone(global_rack_id, general_labor_racks),
                        'confidence': 0.9,
                        'description': f"Rack moved from powder_booth to general_labor",
                        'timestamp': _now_iso()
                    })
                else:
                    # Rack disappeared entirely
                    movements.append({
                        'type': 'disappearance',
                        'global_rack_id': global_rack_id,
                        'from_camera': 'powder_booth',
                        'from_zone': rack_state.get('current_zone', 'unknown'),
                        'confidence': 0.7,
                        'description': f"Rack disappeared from powder_booth",
                        'timestamp': _now_iso()
                    })
        
        return movements
    
    def _handle_dual_appearances(self, general_labor_racks: List[Dict], 
                               powder_booth_racks: List[Dict], current_state: Dict) -> List[Dict]:
        """
        Handle racks that appear in both cameras simultaneously (transition state).
        """
        movements = []
        
        # Find racks detected in both cameras
        gl_global_ids = {r.get('global_rack_id') for r in general_labor_racks if r.get('global_rack_id')}
        pb_global_ids = {r.get('global_rack_id') for r in powder_booth_racks if r.get('global_rack_id')}
        
        dual_racks = gl_global_ids.intersection(pb_global_ids)
        
        for global_rack_id in dual_racks:
            if not global_rack_id:
                continue
                
            # This rack appears in both cameras - it's in transition
            gl_rack = next((r for r in general_labor_racks if r.get('global_rack_id') == global_rack_id), None)
            pb_rack = next((r for r in powder_booth_racks if r.get('global_rack_id') == global_rack_id), None)
            
            if gl_rack and pb_rack:
                # Determine direction of movement based on zones
                gl_zone = gl_rack.get('zone_description', '')
                pb_zone = pb_rack.get('zone_description', '')
                
                # Check which zone suggests the rack is leaving vs arriving
                movement_direction = self._determine_movement_direction(gl_zone, pb_zone)
                
                movements.append({
                    'type': 'transition_state',
                    'global_rack_id': global_rack_id,
                    'gl_zone': gl_zone,
                    'pb_zone': pb_zone,
                    'direction': movement_direction,
                    'confidence': 0.8,
                    'description': f"Rack in transition between cameras (direction: {movement_direction})",
                    'timestamp': _now_iso()
                })
                
                logger.info(f"Dual appearance: {global_rack_id} in both cameras ({movement_direction})")
        
        return movements
    
    def _get_detected_zone(self, global_rack_id: str, rack_list: List[Dict]) -> str:
        """Get the zone where a specific rack was detected."""
        for rack in rack_list:
            if rack.get('global_rack_id') == global_rack_id:
                return rack.get('zone_description', 'unknown')
        return 'unknown'
    
    def _determine_movement_direction(self, gl_zone: str, pb_zone: str) -> str:
        """
        Determine direction of movement based on zone positions.
        """
        # General labor transition zones
        gl_exit_zones = ['right_side', 'center_right']
        gl_entry_zones = ['left_side', 'center_left']
        
        # Powder booth zones
        pb_entry_zones = ['entrance']
        pb_exit_zones = ['entrance']  # Same zone for entry/exit in powder booth
        
        # Analyze zone combinations to determine direction
        if gl_zone in gl_exit_zones and pb_zone in pb_entry_zones:
            return "general_labor_to_powder_booth"
        elif gl_zone in gl_entry_zones and pb_zone in pb_exit_zones:
            return "powder_booth_to_general_labor"
        else:
            return "undetermined"
    
    def process_detected_movements(self, movements: List[Dict]) -> None:
        """
        Process and persist detected movements to database.
        """
        try:
            for movement in movements:
                movement_type = movement.get('type')
                global_rack_id = movement.get('global_rack_id')
                
                if movement_type == 'cross_zone_movement':
                    # This is a confirmed movement - update database
                    add_rack_movement(
                        global_rack_id=global_rack_id,
                        from_camera=movement['from_camera'],
                        to_camera=movement['to_camera'],
                        from_zone=movement['from_zone'],
                        to_zone=movement['to_zone'],
                        detection_method="cross_zone_correlation",
                        confidence=movement['confidence'],
                        notes=movement['description']
                    )
                    
                    # Update rack's current position
                    update_global_rack_position(
                        rack_id=global_rack_id,
                        camera_id=movement['to_camera'],
                        zone=movement['to_zone'],
                        position_description=f"{movement['to_zone']} in {movement['to_camera']}"
                    )
                    
                    logger.info(f"Processed movement: {global_rack_id} from {movement['from_camera']} to {movement['to_camera']}")
                
                elif movement_type == 'transition_state':
                    # Mark rack as in transit
                    update_global_rack_position(
                        rack_id=global_rack_id,
                        camera_id='in_transit',
                        zone=f"between_{movement['gl_zone']}_{movement['pb_zone']}",
                        position_description=f"In transition: {movement['description']}"
                    )
                    
                    # Store transition for future resolution
                    self.pending_transitions[global_rack_id] = {
                        'timestamp': movement['timestamp'],
                        'gl_zone': movement['gl_zone'],
                        'pb_zone': movement['pb_zone'],
                        'direction': movement['direction']
                    }
                    
                    logger.info(f"Marked {global_rack_id} as in transition")
                
                elif movement_type == 'disappearance':
                    # Handle rack disappearance - could be temporary
                    logger.warning(f"Rack {global_rack_id} disappeared from {movement['from_camera']}")
                    # Don't immediately mark as missing - might reappear
                
                elif movement_type == 'unexpected_appearance':
                    # This might indicate a missed movement or detection error
                    logger.warning(f"Unexpected appearance: {movement['description']}")
                    
                    # Update position but note the anomaly
                    update_global_rack_position(
                        rack_id=global_rack_id,
                        camera_id=movement['to_camera'],
                        zone=movement['to_zone'],
                        position_description=f"Unexpected appearance in {movement['to_zone']}"
                    )
        
        except Exception as e:
            logger.error(f"Failed to process detected movements: {e}")
    
    def resolve_pending_transitions(self) -> int:
        """
        Resolve pending transitions that have been in progress too long.
        
        Returns:
            Number of transitions resolved
        """
        try:
            resolved_count = 0
            current_time = datetime.now()
            transition_timeout = timedelta(minutes=TRANSITION_OVERLAP_MINUTES)
            
            expired_transitions = []
            
            for global_rack_id, transition_data in self.pending_transitions.items():
                transition_time = datetime.fromisoformat(transition_data['timestamp'].replace('Z', ''))
                
                if current_time - transition_time > transition_timeout:
                    expired_transitions.append(global_rack_id)
                    
                    # Try to determine final position
                    direction = transition_data.get('direction', 'undetermined')
                    
                    if direction == "general_labor_to_powder_booth":
                        # Assume rack completed move to powder booth
                        update_global_rack_position(
                            rack_id=global_rack_id,
                            camera_id='powder_booth',
                            zone=transition_data['pb_zone'],
                            position_description=f"Completed transition to powder_booth"
                        )
                        logger.info(f"Resolved transition: {global_rack_id} → powder_booth")
                        
                    elif direction == "powder_booth_to_general_labor":
                        # Assume rack completed move to general labor
                        update_global_rack_position(
                            rack_id=global_rack_id,
                            camera_id='general_labor',
                            zone=transition_data['gl_zone'],
                            position_description=f"Completed transition to general_labor"
                        )
                        logger.info(f"Resolved transition: {global_rack_id} → general_labor")
                        
                    else:
                        # Undetermined direction - mark as missing
                        update_global_rack_position(
                            rack_id=global_rack_id,
                            camera_id=None,
                            zone=None,
                            position_description=f"Transition timeout - location unknown"
                        )
                        logger.warning(f"Transition timeout: {global_rack_id} location unknown")
                    
                    resolved_count += 1
            
            # Remove resolved transitions
            for rack_id in expired_transitions:
                del self.pending_transitions[rack_id]
            
            return resolved_count
            
        except Exception as e:
            logger.error(f"Failed to resolve pending transitions: {e}")
            return 0
    
    def get_transition_summary(self) -> Dict:
        """
        Get summary of current transition states and recent movements.
        
        Returns:
            Dict with transition statistics and status
        """
        try:
            recent_movements = get_recent_movements(hours=4)
            
            # Count movements by direction
            gl_to_pb = len([m for m in recent_movements if m['from_camera'] == 'general_labor' and m['to_camera'] == 'powder_booth'])
            pb_to_gl = len([m for m in recent_movements if m['from_camera'] == 'powder_booth' and m['to_camera'] == 'general_labor'])
            
            # Get pending transitions info
            pending_count = len(self.pending_transitions)
            pending_details = []
            
            current_time = datetime.now()
            for rack_id, transition in self.pending_transitions.items():
                transition_time = datetime.fromisoformat(transition['timestamp'].replace('Z', ''))
                duration_minutes = (current_time - transition_time).total_seconds() / 60
                
                pending_details.append({
                    'rack_id': rack_id,
                    'direction': transition['direction'],
                    'duration_minutes': round(duration_minutes, 1),
                    'gl_zone': transition['gl_zone'],
                    'pb_zone': transition['pb_zone']
                })
            
            return {
                'recent_movements': {
                    'total': len(recent_movements),
                    'general_labor_to_powder_booth': gl_to_pb,
                    'powder_booth_to_general_labor': pb_to_gl
                },
                'pending_transitions': {
                    'count': pending_count,
                    'details': pending_details
                },
                'summary_timestamp': _now_iso()
            }
            
        except Exception as e:
            logger.error(f"Failed to get transition summary: {e}")
            return {
                'recent_movements': {'total': 0, 'general_labor_to_powder_booth': 0, 'powder_booth_to_general_labor': 0},
                'pending_transitions': {'count': 0, 'details': []},
                'error': str(e),
                'summary_timestamp': _now_iso()
            }
    
    def validate_movement_patterns(self) -> Dict:
        """
        Validate recent movement patterns to detect anomalies.
        
        Returns:
            Dict with validation results and recommendations
        """
        try:
            recent_movements = get_recent_movements(hours=24)
            
            issues = []
            recommendations = []
            
            # Check for excessive movements
            movement_counts = {}
            for movement in recent_movements:
                rack_id = movement['global_rack_id']
                movement_counts[rack_id] = movement_counts.get(rack_id, 0) + 1
            
            for rack_id, count in movement_counts.items():
                if count > 8:  # More than 8 movements in 24 hours
                    issues.append(f"Rack {rack_id} moved {count} times in 24 hours")
                    recommendations.append(f"Check if {rack_id} detection is stable or if rack is actually moving frequently")
            
            # Check for rapid back-and-forth movements
            for rack_id in movement_counts:
                rack_movements = [m for m in recent_movements if m['global_rack_id'] == rack_id]
                rack_movements.sort(key=lambda x: x['movement_timestamp'])
                
                rapid_reversals = 0
                for i in range(1, len(rack_movements)):
                    prev_move = rack_movements[i-1]
                    curr_move = rack_movements[i]
                    
                    # Check if movement reversed direction quickly
                    if (prev_move['from_camera'] == curr_move['to_camera'] and 
                        prev_move['to_camera'] == curr_move['from_camera']):
                        
                        prev_time = datetime.fromisoformat(prev_move['movement_timestamp'].replace('Z', ''))
                        curr_time = datetime.fromisoformat(curr_move['movement_timestamp'].replace('Z', ''))
                        
                        if (curr_time - prev_time).total_seconds() < 3600:  # Less than 1 hour
                            rapid_reversals += 1
                
                if rapid_reversals > 2:
                    issues.append(f"Rack {rack_id} has {rapid_reversals} rapid direction reversals")
                    recommendations.append(f"Check detection accuracy for {rack_id} - may be false movements")
            
            # Check for stalled transitions
            stalled_count = 0
            for rack_id, transition in self.pending_transitions.items():
                transition_time = datetime.fromisoformat(transition['timestamp'].replace('Z', ''))
                if (datetime.now() - transition_time).total_seconds() > 1800:  # 30 minutes
                    stalled_count += 1
            
            if stalled_count > 0:
                issues.append(f"{stalled_count} racks have been in transition for over 30 minutes")
                recommendations.append("Review transition detection logic or manually resolve stalled transitions")
            
            return {
                'validation_timestamp': _now_iso(),
                'total_issues': len(issues),
                'issues': issues,
                'recommendations': recommendations,
                'movement_statistics': {
                    'total_movements_24h': len(recent_movements),
                    'active_movers': len(movement_counts),
                    'pending_transitions': len(self.pending_transitions)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to validate movement patterns: {e}")
            return {
                'validation_timestamp': _now_iso(),
                'total_issues': 1,
                'issues': [f"Validation failed: {str(e)}"],
                'recommendations': ["Check database connectivity and movement tracking system"],
                'movement_statistics': {}
            }


# Global instance for use by other modules
rack_transition_manager = RackTransitionManager()