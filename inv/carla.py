import json
import requests
import sys
import time
from typing import Any, Dict, List, Optional
import openai
sys.path.append('../')


class JSONPromptBuilder:
    def __init__(self):
        self.base_template = """
You are CARLA's affordance prediction system for AI2-THOR environments. Analyze the scene data and predict the most likely next actions the user will take.

CURRENT SCENE CONTEXT:
{scene_context}

FOCUSED OBJECT:
{focused_object_json}

NEARBY OBJECTS:
{nearby_objects_json}

AGENT STATE:
{agent_state_json}

RECENT ACTION HISTORY:
{history_buffer}

USER PROFILE:
{user_embedding}

AVAILABLE AFFORDANCE PRIMITIVES:
- grasp: pick up an object
- place: put down a held object  
- slice: cut an object with a knife
- sip: drink from a container
- fill: add liquid to a container
- open: open a door/drawer/container
- close: close a door/drawer/container
- toggle: turn on/off a toggleable object
- cook: heat an object
- clean: wash a dirty object
- eat: consume food
- move: relocate an object
- pour: empty liquid from container
- noop: no operation (do nothing)

OUTPUT FORMAT:
Return exactly 5 affordance primitives ranked by probability as a JSON array with each object containing
primitive, confidence (0.0 to 1.0), target (objectId), and description of the affordance:
[
  {{"primitive": "<primitive>", "confidence": <confidence>, "target": "<target>", "description": "<description>"}},
  {{"primitive": "<primitive>", "confidence": <confidence>, "target": "<target>", "description": "<description>"}},
  {{"primitive": "<primitive>", "confidence": <confidence>, "target": "<target>", "description": "<description>"}},
  {{"primitive": "<primitive>", "confidence": <confidence>, "target": "<target>", "description": "<description>"}},
  {{"primitive": "<primitive>", "confidence": <confidence>, "target": "<target>", "description": "<description>"}}
]

REASONING GUIDELINES:
1. Consider object states (pickupable, sliceable, openable, etc.)
2. Factor in spatial relationships (distance, height, orientation)
3. Account for agent's recent actions and success/failure
4. Prioritize immediate, atomic actions over complex sequences
5. Consider user's personal interaction patterns
6. Ensure target object is specified for each affordance
7. Confidences must sum to 1.0

Respond with ONLY the JSON array, no additional text.
"""

    def build_prompt(self, 
                    thor_event: Any,
                    selected_object: Optional[Dict] = None, 
                    nearby_objects: Optional[List[Dict]] = None,
                    history_buffer: Optional[List[Dict]] = None,
                    user_embedding: Optional[str] = None) -> str:
        """Build complete prompt from AI2-THOR event data"""
        
        # Extract scene context
        scene_context = self._format_scene_context(thor_event.metadata)
        
        # Format focused object (clean up the JSON for readability)
        focused_object_json = self._format_focused_object(selected_object)
        
        # Format nearby objects (limit to most relevant)
        nearby_objects_json = self._format_nearby_objects(nearby_objects, thor_event.metadata.get('objects', []))
        
        # Format agent state
        agent_state_json = self._format_agent_state(thor_event.metadata.get('agent', {}), thor_event.metadata)
        
        # Format history
        history_text = self._format_history(history_buffer)
        
        # Format user profile
        user_text = "User only likes sliced apples" 
        
        return self.base_template.format(
            scene_context=scene_context,
            focused_object_json=focused_object_json,
            nearby_objects_json=nearby_objects_json,
            agent_state_json=agent_state_json,
            history_buffer=history_text,
            user_embedding=user_text
        )
    
    def _format_scene_context(self, metadata: Dict) -> str:
        """Extract high-level scene information"""
        context = {
            "scene_name": metadata.get('sceneName', 'Unknown'),
            "scene_at_rest": metadata.get('isSceneAtRest', True),
            "agent_collided": metadata.get('collided', False),
            "last_action": metadata.get('lastAction', 'None'),
            "last_action_success": metadata.get('lastActionSuccess', True),
            "error_message": metadata.get('errorMessage', ''),
            "visible_objects_count": len([obj for obj in metadata.get('objects', []) if obj.get('visible', False)])
        }
        return json.dumps(context, indent=2)
    
    def _format_focused_object(self, selected_object: Optional[Dict]) -> str:
        """Format the main object of interest, filtering relevant properties"""
        if not selected_object:
            return "null"
        
        # Filter to most relevant properties for affordance reasoning
        relevant_props = {
            "name": selected_object.get('name'),
            "objectType": selected_object.get('objectType'),
            "objectId": selected_object.get('objectId'),
            "position": selected_object.get('position'),
            "rotation": selected_object.get('rotation'),
            "distance": selected_object.get('distance'),
            
            # Interaction capabilities
            "visible": selected_object.get('visible'),
            "isInteractable": selected_object.get('isInteractable'),
            "pickupable": selected_object.get('pickupable'),
            "isPickedUp": selected_object.get('isPickedUp'),
            
            # Physical states
            "breakable": selected_object.get('breakable'),
            "isBroken": selected_object.get('isBroken'),
            "sliceable": selected_object.get('sliceable'),
            "isSliced": selected_object.get('isSliced'),
            "cookable": selected_object.get('cookable'),
            "isCooked": selected_object.get('isCooked'),
            
            # Container properties
            "openable": selected_object.get('openable'),
            "isOpen": selected_object.get('isOpen'),
            "openness": selected_object.get('openness'),
            "canFillWithLiquid": selected_object.get('canFillWithLiquid'),
            "isFilledWithLiquid": selected_object.get('isFilledWithLiquid'),
            "fillLiquid": selected_object.get('fillLiquid'),
            
            # Other states
            "toggleable": selected_object.get('toggleable'),
            "isToggled": selected_object.get('isToggled'),
            "dirtyable": selected_object.get('dirtyable'),
            "isDirty": selected_object.get('isDirty'),
            "canBeUsedUp": selected_object.get('canBeUsedUp'),
            "isUsedUp": selected_object.get('isUsedUp'),
            
            # Physical properties
            "temperature": selected_object.get('temperature'),
            "mass": selected_object.get('mass'),
            "salientMaterials": selected_object.get('salientMaterials'),
            "moveable": selected_object.get('moveable'),
            "isMoving": selected_object.get('isMoving'),
            
            # Receptacle info
            "receptacle": selected_object.get('receptacle'),
            "receptacleObjectIds": selected_object.get('receptacleObjectIds'),
            "parentReceptacles": selected_object.get('parentReceptacles')
        }
        
        # Remove None values for cleaner JSON
        relevant_props = {k: v for k, v in relevant_props.items() if v is not None}
        
        return json.dumps(relevant_props, indent=2)
    
    def _format_nearby_objects(self, nearby_objects: Optional[List[Dict]], all_objects: List[Dict]) -> str:
        """Format nearby objects, limiting to most relevant"""
        if nearby_objects:
            objects_to_use = nearby_objects
        else:
            # Auto-select nearby visible interactable objects
            objects_to_use = [
                obj for obj in all_objects 
                if obj.get('visible', False) and 
                   obj.get('isInteractable', False) and 
                   obj.get('distance', float('inf')) < 5.0
            ]
            # Limit to 5 closest objects
            objects_to_use = sorted(objects_to_use, key=lambda x: x.get('distance', 0))[:5]
        
        if not objects_to_use:
            return "[]"
        
        # Simplified object info for context
        simplified_objects = []
        for obj in objects_to_use:
            simplified = {
                "objectType": obj.get('objectType'),
                "objectId": obj.get('objectId'),
                "distance": obj.get('distance'),
                "pickupable": obj.get('pickupable'),
                "openable": obj.get('openable'),
                "isOpen": obj.get('isOpen'),
                "receptacle": obj.get('receptacle')
            }
            # Remove None values
            simplified = {k: v for k, v in simplified.items() if v is not None}
            simplified_objects.append(simplified)
        
        return json.dumps(simplified_objects, indent=2)
    
    def _format_agent_state(self, agent_data: Dict, metadata: Dict) -> str:
        """Format agent state information"""
        agent_state = {
            "position": agent_data.get('position'),
            "rotation": agent_data.get('rotation'),
            "cameraPosition": metadata.get('cameraPosition'),
            "cameraRotation": metadata.get('cameraRotation'),
            "last_action": metadata.get('lastAction'),
            "last_action_success": metadata.get('lastActionSuccess'),
            "collided": metadata.get('collided'),
            "held_objects": metadata.get('inventoryObjects', []),
            "agent_id": metadata.get('agentId')
        }
        
        # Remove None values
        agent_state = {k: v for k, v in agent_state.items() if v is not None}
        
        return json.dumps(agent_state, indent=2)
    
    def _format_history(self, history_buffer: Optional[List[Dict]]) -> str:
        """Format recent action history"""
        if not history_buffer:
            return "No recent actions recorded"
        
        history_lines = []
        for i, entry in enumerate(history_buffer[-5:]):  # Last 5 actions
            action = entry.get('action', 'Unknown')
            success = entry.get('success', False)
            reward = entry.get('reward', 0.0)
            target = entry.get('target', 'Unknown')
            
            history_lines.append(f"{i+1}. {action} on {target} - Success: {success}, Reward: {reward:.2f}")
        
        return "\n".join(history_lines)
    
    def _format_user_profile(self, user_embedding: Optional[str]) -> str:
        """Format user personalization information"""
        if not user_embedding:
            return "No user profile available"
        
        return f"User ID: {user_embedding}\nPersonalization: Active learning from user preferences"

    def parse_llm_response(self, response_text: str) -> Optional[List[Dict]]:
        """Parse LLM JSON response into structured affordances"""
        try:
            # Extract JSON from response (in case there's extra text)
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                print("No JSON array found in response")
                return None
            
            json_text = response_text[start_idx:end_idx]
            affordances = json.loads(json_text)
            
            # Validate structure
            if not isinstance(affordances, list) or len(affordances) != 5:
                print(f"Invalid affordance format: expected list of 5, got {type(affordances)} of length {len(affordances) if isinstance(affordances, list) else 'N/A'}")
                return None
            
            # Validate each affordance
            for i, aff in enumerate(affordances):
                required_keys = ['primitive', 'confidence', 'target', 'description']
                if not all(key in aff for key in required_keys):
                    print(f"Affordance {i} missing required keys: {required_keys}")
                    return None
            
            # Normalize confidences to sum to 1.0
            total_confidence = sum(aff['confidence'] for aff in affordances)
            if total_confidence > 0:
                for aff in affordances:
                    aff['confidence'] = aff['confidence'] / total_confidence
            
            return affordances
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            return None
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return None
        
class JSONAffordanceLLM:
    def __init__(self, model="gpt-4", ollama_url=None, temperature=0.7):
        self.model = model
        self.ollama_url = ollama_url
        self.temperature = temperature
        # OpenAI
        self.client = openai.OpenAI()

    
    def get_affordances(self, prompt: str) -> Optional[List[Dict]]:
        """Get affordance predictions using JSON format"""
        try:
            if self.ollama_url: 
                # Use Ollama
                data = {
                    "model": "llama3",
                    "messages": [
                        {"role": "system", "content": "You are an expert affordance prediction system. Always respond with valid JSON arrays only."},
                        {
                            "role": "user",
                            "content": prompt

                        }
                    ],
                    "stream": False,
                }

                headers = {
                    "Content-Type": "application/json"
                }

                response_text = requests.post(self.ollama_url, headers=headers, json=data)
            else:
                # Use OpenAI
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert affordance prediction system. Always respond with valid JSON arrays only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
            
                response_text = response.choices[0].message.content

            # Convert content to JSON affordances
            parser = JSONPromptBuilder()
            return parser.parse_llm_response(response_text.json()["message"]["content"])

            
        except Exception as e:
            print(f"LLM API error: {e}")
            return self._fallback_affordances()
    
    def _fallback_affordances(self) -> List[Dict]:
        """Fallback affordances if LLM fails"""
        return None
        # [
        #     {"primitive": "grasp", "confidence": 0.4, "target": "Unknown", "description": "Pick up object"},
        #     {"primitive": "place", "confidence": 0.3, "target": "Unknown", "description": "Put down object"},
        #     {"primitive": "open", "confidence": 0.1, "target": "Unknown", "description": "Open container"},
        #     {"primitive": "close", "confidence": 0.1, "target": "Unknown", "description": "Close container"},
        #     {"primitive": "move", "confidence": 0.1, "target": "Unknown", "description": "Relocate object"}
        # ]
    
class Carla:
    def __init__(self, ollama_url: str = None):
        self.prompt_builder = JSONPromptBuilder()
        self.llm = JSONAffordanceLLM(ollama_url=ollama_url)
    
    def get_affordances(self, 
                        thor_event,
                        selected_object,
                        user_vector=None,
                        history_buffer=None) -> Optional[Dict]:
        """Complete pipeline using JSON approach"""
        
        # Build prompt with JSON data
        prompt = self.prompt_builder.build_prompt(
            thor_event=thor_event,
            selected_object=selected_object,
            nearby_objects=None,  # Auto-detected
            history_buffer=history_buffer,
            user_embedding=str(user_vector) if user_vector else None
        )

        print("Generated Prompt:\n", prompt)
        
        print("Getting affordances from LLM...")
        
        # Get LLM predictions
        affordances = self.llm.get_affordances(prompt)
        
        if not affordances:
            return None
        
        # Convert to CARLA-compatible tree format
        tree = {
            'root': {
                'children': [
                    {
                        'primitive': aff['primitive'],
                        'probability': aff['confidence'],
                        'target': aff['target'],
                        'description': aff['description'],
                        'children': []
                    }
                    for aff in affordances
                ]
            },
            'metadata': {
                'thor_scene': thor_event.metadata.get('sceneName'),
                'selected_object_id': selected_object.get('objectId') if selected_object else None,
                'approach': 'json',
                'timestamp': time.time()
            }
        }
        
        return tree