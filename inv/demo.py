import sys
sys.path.append('../')
import time
from typing import Any, Dict
from ai2thor.controller import Controller
import ai2thor
import matplotlib.pyplot as plt
import numpy as np
from carla import Carla

def main():
    print("Starting AI2-THOR Controller...")
    controller = Controller(
        agentMode='neural_os',
        server_class=ai2thor.wsgi_server.WsgiServer,
        host="192.168.86.47",
        port=8200,
        start_unity=False,
    ) 

    print("Controller started successfully.")

    # Run loop that checks metadata every one second
    print("Starting...")
    carla = Carla(ollama_url="http://localhost:11434/api/chat",)
    selected_object = None
    in_progress = False

    # controller.step("UnpausePhysicsAutoSim")

    while True:
        # try:
        # Get new event from controller
        event = controller.step(action='Done')
        metadata = event.metadata
        # print(metadata)
        
        # Check to see if user has selected object
        if 'selectedObject' in metadata['neuralOS'] and not in_progress:
            if metadata['neuralOS']['selectedObject'] != None and (selected_object == None or selected_object['objectId'] != metadata['neuralOS']['selectedObject']['objectId']):
                in_progress = True
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

                    if selected_affordance['primitive'].lower() == 'slice':
                        print("Slicing...")
                        controller.step(action="RotateLeft")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="RotateLeft")
                        controller.step("LookDown")
                        event = controller.step(action="SliceObject", objectId=selected_object['objectId'])
                        controller.step(action="PickupObject", objectId='Apple|-00.49|+01.15|+00.52|AppleSliced_0')
                        controller.step("LookUp")
                        controller.step(action="RotateLeft")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="RotateRight")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="DropHandObject")

                        controller.step(action="Done")

                        print("Slicing successfully!")
                        in_progress = False
                    elif selected_affordance['primitive'].lower() == 'grasp':
                        print("Grasping...")
                        controller.step(action="RotateLeft")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="RotateLeft")
                        controller.step("LookDown")
                        controller.step(action="PickupObject", objectId=selected_object['objectId'])
                        controller.step("LookUp")
                        controller.step(action="RotateLeft")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="RotateRight")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="MoveAhead")
                        controller.step(action="DropHandObject")

                        controller.step(action="Done")

                        print("Grasped successfully!")
                        in_progress = False

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


if __name__ == "__main__":
    main()
    print("Exiting...")
    sys.exit(0)