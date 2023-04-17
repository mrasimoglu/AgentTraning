using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Academy : MonoBehaviour
{
    [System.Serializable]
    public class Classroom
    {
        public int repeatCount;
        public Unity.MLAgents.Agent[] agents;
        public Environment[] environments;
    }

    public int distanceBetweenEnvironments;
    public Classroom[] classrooms;

    // Start is called before the first frame update
    [System.Obsolete]
    void Start()
    {
        for (int i_class = 0; i_class < classrooms.Length; ++i_class)
        {
            GameObject class_obj = new GameObject("Class Group " + (i_class + 1).ToString());
            class_obj.transform.SetParent(this.transform);
            for (int i_repeat = 0; i_repeat < classrooms[i_class].repeatCount; ++i_repeat)
            {
                GameObject class_obj2 = new GameObject("Class " + (i_repeat + 1).ToString());
                class_obj2.transform.position = new Vector3(i_repeat * distanceBetweenEnvironments, i_class * distanceBetweenEnvironments, 0);
                class_obj2.transform.SetParent(class_obj.transform);

                foreach (var agent in classrooms[i_class].agents)
                {
                    GameObject agnt = Instantiate(agent.gameObject) as GameObject;
                    agnt.transform.SetParent(class_obj2.transform, false);
                }

                foreach (var environment in classrooms[i_class].environments)
                {
                    GameObject env = Instantiate(environment.gameObject) as GameObject;
                    env.transform.SetParent(class_obj2.transform, false);
                }
            }
        }
    }

    // Update is called once per frame
    void Update()
    {

    }
}
