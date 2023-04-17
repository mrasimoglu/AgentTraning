using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using System;
using System.Threading;
using UnityEngine.SceneManagement;
using System.Linq;

public class Drone : Agent
{

    private Rigidbody rigidbody;
    private static bool flag = false;

    private float episode_reward = 0;

    private int step;

    private List<GameObject> contacts = new List<GameObject>();
    List<GameObject> checkpoints = new List<GameObject>();
    List<GameObject> soldiers = new List<GameObject>();

    private float timeSinceDecision;
    public float timeBetweenDecisionsAtInference = 0.2f;

    public int epoch = 0;
    public int max_step;
    public int CheckPointIdx;
    
    Vector3 spawn_pos = new Vector3();
    Vector3 target_pos = new Vector3();

    // Start is called before the first frame update
    void Start()
    {
        rigidbody = GetComponent<Rigidbody>();
        CheckPointIdx = 0;

        List<GameObject> rootObjects = new List<GameObject>();

        Scene scene = SceneManager.GetActiveScene();
        scene.GetRootGameObjects(rootObjects);
        GameObject rootSoldiers = GameObject.Find("soldiers");
        for (int i = 0; i < rootSoldiers.transform.childCount; ++i)
        {
            soldiers.Add(rootSoldiers.transform.GetChild(i).gameObject);
        }

        GameObject rootCheckpoint = rootObjects.Where(x => x.name.ToLower() == "checkpoints").FirstOrDefault();
        for (int i = 0; i < rootCheckpoint.transform.childCount; ++i)
        {
            checkpoints.Add(rootCheckpoint.transform.GetChild(i).gameObject);
        }
    }

    // Update is called once per frame  
    void Update()
    {
    }

    public void FixedUpdate()
    {
        if (timeSinceDecision >= timeBetweenDecisionsAtInference)
        {
            timeSinceDecision = 0f;
            RequestDecision();
        }
        else
        {
            timeSinceDecision += Time.fixedDeltaTime;
        }

        foreach (var contact in contacts)
        {
            Debug.DrawLine(rigidbody.position, contact.GetComponent<Collider>().ClosestPoint(rigidbody.position), Color.red);
        }

        Debug.DrawLine(rigidbody.position, target_pos, Color.green);
    }


    public override void OnEpisodeBegin()
    {
        contacts.Clear();

        for (int i = 0; i < soldiers.Count; ++i)
        {
            soldiers[i].SetActive(true);
            soldiers[i].transform.rotation = Quaternion.Euler(0, soldiers[i].GetComponent<Soldier>().minRotation, 0);
        }

        CheckPointIdx = 0;
        spawn_pos = checkpoints[CheckPointIdx].transform.position;
        target_pos = checkpoints[CheckPointIdx + 1].transform.position;

        rigidbody.angularVelocity = Vector3.zero;
        rigidbody.velocity = Vector3.zero;

        transform.position = spawn_pos;
        transform.rotation = Quaternion.Euler(0, 0, 0);

        step = 0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation((target_pos - rigidbody.position).normalized);

        Vector3 contacts_total = new Vector3(0, 0, 0);
        for (int i = 0; i < 8 && i < contacts.Count; i++)
        {
            Vector3 contact_point = contacts[i].GetComponent<Collider>().ClosestPoint(rigidbody.position);
            Vector3 direction = (contact_point - rigidbody.position);
            contacts_total += direction;
        }
        sensor.AddObservation(contacts_total);
        sensor.AddObservation(rigidbody.velocity);
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        if (flag)
        {
            flag = false;
            EndEpisode();
            epoch++;
            episode_reward = 0;
            return;
        }

        float reward = 0;

        if (step > max_step)
        {
            SetReward(-500);
            flag = true;
            return;
        }

        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = vectorAction[0];
        controlSignal.y = vectorAction[1];
        controlSignal.z = vectorAction[2];
        controlSignal = controlSignal / 5;
        rigidbody.AddForce(controlSignal);

        Debug.DrawLine(rigidbody.position, rigidbody.position + new Vector3(vectorAction[0], vectorAction[1], vectorAction[2]), Color.blue);

        float dist = Vector3.Distance(target_pos, rigidbody.position);
        float initial_dist = Vector3.Distance(spawn_pos, target_pos);

        Vector3 vec = target_pos - rigidbody.position;
        float angle = Vector3.Angle(vec.normalized, rigidbody.velocity.normalized);
        if (angle != 0)
        {
            reward += 3 * (1 - (angle / 90)) * rigidbody.velocity.magnitude;
        }
        else
        {
            reward += 3 * rigidbody.velocity.magnitude;
        }

        for (int i = 0; i < contacts.Count && i < 8; i++)
        {
            Vector3 contact_point = contacts[i].GetComponent<Collider>().ClosestPoint(rigidbody.position);
            vec = contact_point - rigidbody.position;
            angle = Vector3.Angle(vec.normalized, rigidbody.velocity.normalized);

            if (angle != 0)
            {
                reward -= (1 - (angle / 90)) * (rigidbody.velocity.magnitude);
            }
            else
            {
                reward -= 1 * (rigidbody.velocity.magnitude);
            }
        }

        step++;

        reward = reward / 5;

        reward += -(dist / initial_dist) + 0.5f;

        if (Vector3.Distance(rigidbody.position, target_pos) < 2)
        {
            SetReward(500);
            for (int i = 0; i < soldiers.Count; ++i)
            {
                if (soldiers[i].GetComponent<Soldier>().SoldierID == CheckPointIdx + 1)
                {
                    soldiers[i].SetActive(false);
                }
            }

            CheckPointIdx++;
            if (CheckPointIdx == checkpoints.Count - 1)
            {
                CheckPointIdx = 0;
                reward += 1000;
                flag = true;
            }

            spawn_pos = checkpoints[CheckPointIdx].transform.position;
            target_pos = checkpoints[CheckPointIdx + 1].transform.position;

            reward += 500;
        }
        
        SetReward(reward);

        episode_reward += reward;
    }

    private void OnTriggerEnter(Collider other)
    {
        contacts.Add(other.gameObject);
    }

    private void OnTriggerExit(Collider other)
    {
        contacts.Remove(other.gameObject);
    }

    private void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.name != "DroneTarget")
        {
            SetReward(-500);
            flag = true;

            Debug.Log("Episode: " + epoch + " - Reward: " + episode_reward + " - Step: " + step);
        }

    }

}
