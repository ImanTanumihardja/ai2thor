import sys
sys.path.append('../')
import time
from ai2thor.controller import Controller
import ai2thor
import matplotlib.pyplot as plt
import numpy as np
import heapq
from typing import Dict, List, Optional, Any
import math
from dataclasses import dataclass
from enum import Enum
from carla import Carla

def main():
    print("Starting AI2-THOR Controller...")
    controller = Controller(
        agentMode='neural_os',
        server_class=ai2thor.wsgi_server.WsgiServer,
        host="127.0.0.1",
        port=8200,
        start_unity=False,
    ) 

    print("Controller started successfully.")

    # Initialize the scene
    print("Initializing scene...")
    # controller.step("UnpausePhysicsAutoSim")

    # Run loop that checks metadata every one second
    print("Starting...")
    carla = Carla(ollama_url="http://localhost:11434/api/chat")
    navigator = THORNavigator(controller)
    selected_object = None

    while True:
        # try:
        # Get new event from controller
        event = controller.step(action='Done')
        metadata = event.metadata
        # print(metadata)
        
        # Check to see if user has selected object
        if 'selectedObject' in metadata['neuralOS']:
            if metadata['neuralOS']['selectedObject'] != None and selected_object != metadata['neuralOS']['selectedObject']:
                selected_object = metadata['neuralOS']['selectedObject']

                # Obtain affordances
                affordances = carla.get_affordances(event, selected_object)

                # Sort children by probability
                sorted_children = sorted(affordances['root']['children'], key=lambda x: x.get('probability', 0), reverse=True)

                affordances['root']['children'] = sorted_children

                if affordances:
                    print_affordances(affordances)
                    
                    # Grab the most possible affordance
                    selected_affordance = affordances['root']['children'][0]

                    if selected_affordance['primitive'].lower() == 'grasp':
                        print("Grasping...")
                        result = navigator.navigate_to_object(selected_object)
                        
                        print(f"Navigation result: {result.status.value}")
                        print(f"Message: {result.message}")
                        # controller.step(action="RotateLeft")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="RotateLeft")
                        # controller.step("LookDown")
                        # controller.step(action="PickupObject", objectId=selected_object['objectId'])
                        # controller.step("LookUp")
                        # controller.step(action="RotateLeft")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="RotateRight")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="MoveAhead")
                        # controller.step(action="DropHandObject")
                        # controller.step(action="Done")

                        print("Grasped successfully!")

        # except Exception as e:
        #     print(f"Error during event processing: {e}")
        #     continue
                
        plt.pause(1)

def print_affordances(tree: Dict[str, Any], indent: int = 0) -> None:
    """
    Print affordance tree in a structured, readable format
    
    Args:
        tree: The affordance tree dictionary from CARLA
        indent: Base indentation level (for nested calls)
    """
    
    def print_line(text: str, level: int = 0) -> None:
        """Print a line with proper indentation"""
        spaces = "  " * (indent + level)
        print(f"{spaces}{text}")
    
    # Header
    print_line("🌳 AFFORDANCE TREE")
    print_line("=" * 50)
    print_line("")
    
    # Print metadata if available
    if 'metadata' in tree:
        print_line("📊 METADATA:")
        metadata = tree['metadata']
        
        print_line(f"Scene: {metadata.get('thor_scene', 'Unknown')}", 1)
        print_line(f"Target Object: {metadata.get('selected_object_id', 'Unknown')}", 1)
        print_line(f"Approach: {metadata.get('approach', 'Unknown')}", 1)
        
        # Format timestamp if available
        timestamp = metadata.get('timestamp', 0)
        if timestamp:
            formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
            print_line(f"Timestamp: {formatted_time}", 1)
        
        print_line("")
    
    # Print affordances
    if 'root' in tree and 'children' in tree['root']:
        children = tree['root']['children']
        print_line("🎯 PREDICTED AFFORDANCES:")
        print_line(f"Total Options: {len(children)}", 1)
        print_line("")
        
        for i, affordance in enumerate(children, 1):
            primitive = affordance.get('primitive', 'unknown')
            probability = affordance.get('probability', 0)
            target = affordance.get('target', 'unknown')
            description = affordance.get('description', 'No description')
            
            # Create confidence bar visualization
            confidence_percent = int(probability * 100)
            bar_length = 20
            filled_length = int(bar_length * probability)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            # Print affordance details
            print_line(f"{i}. {primitive.upper()}", 1)
            print_line(f"Target: {target}", 2)
            print_line(f"Confidence: {probability:.3f} ({confidence_percent}%) [{bar}]", 2)
            print_line(f"Description: {description}", 2)
            print_line("")
    
    # Summary statistics
    if 'root' in tree and 'children' in tree['root']:
        children = tree['root']['children']
        
        if children:  # Only if we have affordances
            total_prob = sum(child.get('probability', 0) for child in children)
            max_prob = max((child.get('probability', 0) for child in children))
            min_prob = min((child.get('probability', 0) for child in children))
            
            print_line("📈 SUMMARY STATISTICS:", 0)
            print_line(f"Total Probability: {total_prob:.3f}", 1)
            print_line(f"Highest Confidence: {max_prob:.3f}", 1)
            print_line(f"Lowest Confidence: {min_prob:.3f}", 1)
            
            # Best recommendation
            best_affordance = max(children, key=lambda x: x.get('probability', 0))
            print_line(f"🏆 Best Recommendation: {best_affordance.get('primitive', 'unknown').upper()}", 1)
            print_line("")

@dataclass
class Position:
    """3D Position with utility methods"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)
    
    def distance_2d(self, other: 'Position') -> float:
        """Calculate 2D distance (ignoring Y axis)"""
        return math.sqrt((self.x - other.x)**2 + (self.z - other.z)**2)
    
    def to_dict(self) -> Dict:
        """Convert to THOR-compatible dictionary"""
        return {'x': self.x, 'y': self.y, 'z': self.z}
    
    def __str__(self) -> str:
        return f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

class NavigationStatus(Enum):
    """Navigation result status"""
    SUCCESS = "success"
    FAILED_NO_PATH = "no_path_found"
    FAILED_BLOCKED = "path_blocked"
    FAILED_UNREACHABLE = "target_unreachable"
    FAILED_COLLISION = "collision_detected"
    IN_PROGRESS = "navigating"

@dataclass
class NavigationResult:
    """Result of navigation attempt"""
    status: NavigationStatus
    final_position: Position
    steps_taken: int
    path: List[Position]
    message: str = ""

class THORNavigator:
    """
    Advanced navigation system for AI2-THOR environments
    """
    
    def __init__(self, controller, grid_size: float = 0.25):
        self.controller = controller
        self.grid_size = grid_size
        self.scene_bounds = None
        self.reachable_positions = set()
        self.obstacle_map = {}
        
        # Navigation parameters
        self.max_steps = 100
        self.collision_threshold = 0.3
        self.interaction_distance = 1.5
        self.rotation_step = 90  # degrees
        
        # Initialize scene analysis
        self._analyze_scene()
    
    def _analyze_scene(self):
        """Analyze the current scene to build navigation maps"""
        event = self.controller.last_event
        
        # Get scene bounds
        bounds = event.metadata.get('sceneBounds', {})
        if bounds:
            self.scene_bounds = {
                'min_x': bounds['center']['x'] - bounds['size']['x'] / 2,
                'max_x': bounds['center']['x'] + bounds['size']['x'] / 2,
                'min_z': bounds['center']['z'] - bounds['size']['z'] / 2,
                'max_z': bounds['center']['z'] + bounds['size']['z'] / 2,
                'floor_y': bounds['center']['y'] - bounds['size']['y'] / 2
            }
        
        # Get reachable positions (if available)
        try:
            positions_event = self.controller.step(action="GetReachablePositions")
            if positions_event.metadata['lastActionSuccess']:
                for pos in positions_event.metadata['actionReturn']:
                    self.reachable_positions.add((
                        round(pos['x'] / self.grid_size) * self.grid_size,
                        round(pos['z'] / self.grid_size) * self.grid_size
                    ))
        except:
            print("Warning: Could not get reachable positions, using basic navigation")
        
        # Build obstacle map from scene objects
        self._build_obstacle_map()
    
    def _build_obstacle_map(self):
        """Build map of obstacles from scene objects"""
        event = self.controller.last_event
        self.obstacle_map = {}
        
        for obj in event.metadata['objects']:
            if not obj.get('moveable', False) and obj.get('visible', False):
                # Add object position as obstacle
                pos = obj['position']
                grid_x = round(pos['x'] / self.grid_size) * self.grid_size
                grid_z = round(pos['z'] / self.grid_size) * self.grid_size
                
                # Add some padding around objects
                for dx in [-self.grid_size, 0, self.grid_size]:
                    for dz in [-self.grid_size, 0, self.grid_size]:
                        obstacle_pos = (grid_x + dx, grid_z + dz)
                        self.obstacle_map[obstacle_pos] = obj['objectType']
    
    def navigate_to_object(self, target_object: Dict) -> NavigationResult:
        """
        Navigate to a specific object in the scene
        
        Args:
            target_object: AI2-THOR object dictionary
            
        Returns:
            NavigationResult with status and path information
        """
        target_pos = Position(**target_object['position'])
        
        # Find best approach position near the object
        approach_pos = self._find_best_approach_position(target_object)
        
        if not approach_pos:
            return NavigationResult(
                status=NavigationStatus.FAILED_UNREACHABLE,
                final_position=self._get_current_position(),
                steps_taken=0,
                path=[],
                message=f"No reachable position found near {target_object['objectType']}"
            )
        
        # Navigate to approach position
        return self.navigate_to_position(approach_pos)
    
    def navigate_to_position(self, target_position: Position) -> NavigationResult:
        """
        Navigate to a specific position using A* pathfinding
        
        Args:
            target_position: Target position to reach
            
        Returns:
            NavigationResult with navigation details
        """
        start_pos = self._get_current_position()
        
        print(f"Navigating from {start_pos} to {target_position}")
        
        # Find path using A*
        path = self._find_path(start_pos, target_position)
        
        if not path:
            return NavigationResult(
                status=NavigationStatus.FAILED_NO_PATH,
                final_position=start_pos,
                steps_taken=0,
                path=[],
                message="No valid path found to target"
            )
        
        # Execute the path
        return self._execute_path(path)
    
    def _find_path(self, start: Position, goal: Position) -> List[Position]:
        """
        A* pathfinding algorithm to find optimal path
        
        Args:
            start: Starting position
            goal: Target position
            
        Returns:
            List of positions representing the path
        """
        def heuristic(pos1: Position, pos2: Position) -> float:
            return pos1.distance_2d(pos2)
        
        def get_neighbors(pos: Position) -> List[Position]:
            neighbors = []
            for dx, dz in [(-self.grid_size, 0), (self.grid_size, 0), 
                          (0, -self.grid_size), (0, self.grid_size),
                          (-self.grid_size, -self.grid_size), (-self.grid_size, self.grid_size),
                          (self.grid_size, -self.grid_size), (self.grid_size, self.grid_size)]:
                new_pos = Position(pos.x + dx, pos.y, pos.z + dz)
                
                # Check if position is valid
                if self._is_position_valid(new_pos):
                    neighbors.append(new_pos)
            
            return neighbors
        
        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {self._pos_to_key(start): 0}
        f_score = {self._pos_to_key(start): heuristic(start, goal)}
        closed_set = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            current_key = self._pos_to_key(current)
            
            if current_key in closed_set:
                continue
                
            closed_set.add(current_key)
            
            # Check if we've reached the goal (within grid tolerance)
            if current.distance_2d(goal) < self.grid_size * 1.5:
                # Reconstruct path
                path = []
                while current_key in came_from:
                    path.append(current)
                    current_key = came_from[current_key]
                    current = self._key_to_pos(current_key)
                path.append(start)
                return list(reversed(path))
            
            # Explore neighbors
            for neighbor in get_neighbors(current):
                neighbor_key = self._pos_to_key(neighbor)
                
                if neighbor_key in closed_set:
                    continue
                
                tentative_g = g_score[current_key] + current.distance_2d(neighbor)
                
                if neighbor_key not in g_score or tentative_g < g_score[neighbor_key]:
                    came_from[neighbor_key] = current_key
                    g_score[neighbor_key] = tentative_g
                    f_score[neighbor_key] = tentative_g + heuristic(neighbor, goal)
                    
                    heapq.heappush(open_set, (f_score[neighbor_key], neighbor))
        
        return []  # No path found
    
    def _execute_path(self, path: List[Position]) -> NavigationResult:
        """
        Execute the planned path step by step
        
        Args:
            path: List of positions to visit in order
            
        Returns:
            NavigationResult with execution details
        """
        steps_taken = 0
        executed_path = []
        
        for i, target_pos in enumerate(path[1:], 1):  # Skip starting position
            current_pos = self._get_current_position()
            executed_path.append(current_pos)
            
            # Calculate movement needed
            dx = target_pos.x - current_pos.x
            dz = target_pos.z - current_pos.z
            
            # Determine best action
            if abs(dx) > abs(dz):
                # Move in X direction
                action = "MoveRight" if dx > 0 else "MoveLeft"
            else:
                # Move in Z direction
                action = "MoveAhead" if dz > 0 else "MoveBack"
            
            # Face the correct direction first
            self._face_direction(dx, dz)
            steps_taken += 1
            
            # Execute movement
            event = self.controller.step(action=action)
            steps_taken += 1
            
            if not event.metadata['lastActionSuccess']:
                return NavigationResult(
                    status=NavigationStatus.FAILED_BLOCKED,
                    final_position=self._get_current_position(),
                    steps_taken=steps_taken,
                    path=executed_path,
                    message=f"Movement blocked at step {i}: {event.metadata.get('errorMessage', '')}"
                )
            
            # Check if we've reached max steps
            if steps_taken >= self.max_steps:
                return NavigationResult(
                    status=NavigationStatus.FAILED_BLOCKED,
                    final_position=self._get_current_position(),
                    steps_taken=steps_taken,
                    path=executed_path,
                    message="Maximum steps reached"
                )
        
        final_pos = self._get_current_position()
        executed_path.append(final_pos)
        
        return NavigationResult(
            status=NavigationStatus.SUCCESS,
            final_position=final_pos,
            steps_taken=steps_taken,
            path=executed_path,
            message="Navigation completed successfully"
        )
    
    def _face_direction(self, dx: float, dz: float):
        """Face the agent in the correct direction for movement"""
        if abs(dx) < 0.01 and abs(dz) < 0.01:
            return  # No movement needed
        
        # Calculate target angle
        target_angle = math.degrees(math.atan2(dx, dz))
        
        # Get current rotation
        current_rotation = self.controller.last_event.metadata['agent']['rotation']['y']
        
        # Calculate angle difference
        angle_diff = target_angle - current_rotation
        
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        # Rotate to face direction
        rotation_steps = round(angle_diff / self.rotation_step)
        for _ in range(abs(rotation_steps)):
            if rotation_steps > 0:
                self.controller.step(action="RotateRight")
            else:
                self.controller.step(action="RotateLeft")
    
    def _find_best_approach_position(self, target_object: Dict) -> Optional[Position]:
        """
        Find the best position to approach an object from
        
        Args:
            target_object: AI2-THOR object dictionary
            
        Returns:
            Best approach position or None if no good position found
        """
        target_pos = Position(**target_object['position'])
        
        # Try positions in a circle around the object
        best_pos = None
        best_distance = float('inf')
        
        for angle in range(0, 360, 30):  # Check every 30 degrees
            rad = math.radians(angle)
            
            # Try different distances
            for distance in [self.interaction_distance, self.interaction_distance * 0.8, self.interaction_distance * 1.2]:
                approach_x = target_pos.x + distance * math.cos(rad)
                approach_z = target_pos.z + distance * math.sin(rad)
                approach_pos = Position(approach_x, target_pos.y, approach_z)
                
                # Check if position is valid and reachable
                if self._is_position_reachable(approach_pos):
                    current_pos = self._get_current_position()
                    dist_to_approach = current_pos.distance_2d(approach_pos)
                    
                    if dist_to_approach < best_distance:
                        best_distance = dist_to_approach
                        best_pos = approach_pos
        
        return best_pos
    
    def _is_position_valid(self, pos: Position) -> bool:
        """Check if a position is valid (not in obstacle)"""
        grid_pos = (
            round(pos.x / self.grid_size) * self.grid_size,
            round(pos.z / self.grid_size) * self.grid_size
        )
        
        # Check obstacles
        if grid_pos in self.obstacle_map:
            return False
        
        # Check scene bounds
        if self.scene_bounds:
            if (pos.x < self.scene_bounds['min_x'] or pos.x > self.scene_bounds['max_x'] or
                pos.z < self.scene_bounds['min_z'] or pos.z > self.scene_bounds['max_z']):
                return False
        
        return True
    
    def _is_position_reachable(self, pos: Position) -> bool:
        """Check if a position is reachable by the agent"""
        if not self._is_position_valid(pos):
            return False
        
        # If we have reachable positions, check against them
        if self.reachable_positions:
            grid_pos = (
                round(pos.x / self.grid_size) * self.grid_size,
                round(pos.z / self.grid_size) * self.grid_size
            )
            return grid_pos in self.reachable_positions
        
        # Otherwise assume reachable if valid
        return True
    
    def _get_current_position(self) -> Position:
        """Get agent's current position"""
        agent_pos = self.controller.last_event.metadata['agent']['position']
        return Position(**agent_pos)
    
    def _pos_to_key(self, pos: Position) -> str:
        """Convert position to hashable key"""
        return f"{pos.x:.2f},{pos.z:.2f}"
    
    def _key_to_pos(self, key: str) -> Position:
        """Convert key back to position"""
        x, z = map(float, key.split(','))
        return Position(x, 0.9, z)  # Use standard Y height
    
    # Convenience methods
    def go_to_object_type(self, object_type: str) -> NavigationResult:
        """Navigate to the nearest object of specified type"""
        target_obj = self.find_nearest_object_of_type(object_type)
        if not target_obj:
            return NavigationResult(
                status=NavigationStatus.FAILED_UNREACHABLE,
                final_position=self._get_current_position(),
                steps_taken=0,
                path=[],
                message=f"No {object_type} found in scene"
            )
        
        return self.navigate_to_object(target_obj)
    
    def find_nearest_object_of_type(self, object_type: str) -> Optional[Dict]:
        """Find the nearest object of specified type"""
        current_pos = self._get_current_position()
        nearest_obj = None
        nearest_distance = float('inf')
        
        for obj in self.controller.last_event.metadata['objects']:
            if (obj['objectType'].lower() == object_type.lower() and 
                obj.get('visible', False) and 
                obj.get('isInteractable', False)):
                
                obj_pos = Position(**obj['position'])
                distance = current_pos.distance_2d(obj_pos)
                
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_obj = obj
        
        return nearest_obj
    
    def list_nearby_objects(self, radius: float = 3.0) -> List[Dict]:
        """List all objects within specified radius"""
        current_pos = self._get_current_position()
        nearby_objects = []
        
        for obj in self.controller.last_event.metadata['objects']:
            if obj.get('visible', False):
                obj_pos = Position(**obj['position'])
                distance = current_pos.distance_2d(obj_pos)
                
                if distance <= radius:
                    obj_info = {
                        'object': obj,
                        'distance': distance,
                        'reachable': self._is_position_reachable(obj_pos)
                    }
                    nearby_objects.append(obj_info)
        
        # Sort by distance
        nearby_objects.sort(key=lambda x: x['distance'])
        return nearby_objects
    
    def get_navigation_summary(self) -> Dict:
        """Get summary of navigation capabilities"""
        return {
            'grid_size': self.grid_size,
            'reachable_positions': len(self.reachable_positions),
            'obstacles': len(self.obstacle_map),
            'scene_bounds': self.scene_bounds,
            'current_position': str(self._get_current_position()),
            'max_steps': self.max_steps,
            'interaction_distance': self.interaction_distance
        }

if __name__ == "__main__":
    main()
    print("Exiting...")
    sys.exit(0)