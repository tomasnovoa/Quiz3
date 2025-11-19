using UnityEngine;

public class Planet : MonoBehaviour
{
    public Nave nave;
    public float mass;

    public float influenceRadius;
    bool naveDetectada;

    void Update()
    {
        Vector3 direction = transform.position - nave.transform.position;
        float distance = direction.magnitude;
        if (distance < influenceRadius)
        {
            if (!naveDetectada)
            {
                naveDetectada = true;
                nave.planeta = this;
            }
        }
        else
        {
            if (naveDetectada)
            {
                naveDetectada = false;
                Debug.Log("Se Perdio");
                nave.planeta = null;
            }
        }
    }

    public Vector3 GetGravityForce()
    {
        Vector3 posNave = nave.transform.position;
        Vector3 posPlaneta = transform.position;
        Vector3 direction = posPlaneta - posNave;
        float dist = direction.magnitude;
        float gravityForce = 1 * nave.mass * mass / (dist * dist);
        return direction.normalized * gravityForce;
    }
}
