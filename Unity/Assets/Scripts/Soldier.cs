using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Soldier : MonoBehaviour
{
    public int maxRotation;
    public int minRotation;
    private bool is_increasing;
    public int SoldierID;
    // Start is called before the first frame update
    void Start()
    {
        is_increasing = true;
        this.transform.rotation = Quaternion.Euler(0, minRotation, 0);
    }

    // Update is called once per frame
    void Update()
    {
        this.transform.rotation = is_increasing ? Quaternion.Euler(0, this.transform.rotation.eulerAngles.y + 0.2f, 0) : Quaternion.Euler(0, this.transform.rotation.eulerAngles.y - 0.2f, 0);
        if(this.transform.rotation.eulerAngles.y > maxRotation)
        {
            is_increasing = false;
        }
        if(this.transform.rotation.eulerAngles.y < minRotation)
        {
            is_increasing = true;
        }
    }
}
