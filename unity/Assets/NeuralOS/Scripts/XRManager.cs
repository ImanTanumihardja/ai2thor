
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Events;
using System.Collections;
using TMPro;
using System;
using UnityStandardAssets.Characters.FirstPerson;
using System.Collections.Generic;
using UnityEngine.SceneManagement;
using Unity.XR.CoreUtils;
using Unity.PolySpatial.InputDevices;
using UnityEngine.InputSystem.EnhancedTouch;
using UnityEngine.InputSystem.LowLevel;
using Touch = UnityEngine.InputSystem.EnhancedTouch.Touch;


[RequireComponent(typeof(XROrigin))]
public class XRManager : MonoBehaviour
{
    /// <summary>
    /// The Input Manager assigns callback functions to certain actions that can be perfromed by the XR controllers.
    /// </summary>
    /// 

    private AgentManager _agentManager = null;
    private bool _isInitialized = false;
    private GameObject _selectedObject = null;

    public static XRManager Instance { get; private set; }

    BaseFPSAgentController CurrentActiveController() {
        return _agentManager.PrimaryAgent;
    }

    void OnEnable()
    {
        EnhancedTouchSupport.Enable();
    }

    private void Awake() {
        // If there is an instance, and it's not me, delete myself.
        if (Instance != null && Instance != this) {
            Destroy(Instance.gameObject);
            Instance = this;
        } else {
            Instance = this;
        }

        _agentManager = GameObject.Find("PhysicsSceneManager").GetComponentInChildren<AgentManager>();
    }
        

    public void Initialize() {
        if (_isInitialized) {
            return;
        }
        Dictionary<string, object> action = new Dictionary<string, object>();
        // if you want to use smaller grid size step increments, initialize with a smaller/larger gridsize here
        // by default the gridsize is 0.25, so only moving in increments of .25 will work
        // so the MoveAhead action will only take, by default, 0.25, .5, .75 etc magnitude with the default
        // grid size!
        // action.renderNormalsImage = true;
        // action.renderDepthImage = true;
        // action.renderSemanticSegmentation = true;
        // action.renderInstanceSegmentation = true;
        // action.renderFlowImage = true;
        // action.rotateStepDegrees = 30;
        // action.ssao = "default";
        // action.snapToGrid = true;
        // action.makeAgentsVisible = false;
        action["agentMode"] = "vr";
        action["fieldOfView"] = 90f;
        // action.cameraY = 2.0f;
        action["snapToGrid"] = true;
        // action.rotateStepDegrees = 45;
        action["autoSimulation"] = true;
        action["action"] = "Initialize";
        CurrentActiveController().ProcessControlCommand(new DynamicServerAction(action), _agentManager);

        _isInitialized = true;
    }

    private void Update() {
        if (Input.GetKeyDown(KeyCode.Space)) {
            Initialize();
        }

        var activeTouches = Touch.activeTouches;

        // You can determine the number of active inputs by checking the count of activeTouches
        if (activeTouches.Count > 0)
        {
            var primaryTouchData = EnhancedSpatialPointerSupport.GetPointerState(activeTouches[0]);
            if (activeTouches[0].phase == UnityEngine.InputSystem.TouchPhase.Began)
            {
                // allow balloons to be popped with a poke or indirect pinch
                if (primaryTouchData.Kind == SpatialPointerKind.IndirectPinch || primaryTouchData.Kind == SpatialPointerKind.Touch)
                {
                    var obj = primaryTouchData.targetObject;
                    if (obj != null)
                    {
                        GameObject objParent = obj.transform.parent.gameObject;
                        if (objParent != null && objParent.tag == "SimObjPhysics" && objParent.GetComponent<SimObjPhysics>() != null)
                        {
                            _selectedObject = objParent;
                            Debug.Log("Selected object: " + _selectedObject.name);
                        }
                    }
                }
            }
        }
    }
}
