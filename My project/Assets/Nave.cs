using UnityEngine;

public class Nave : MonoBehaviour
{
    public float speed;
    private Vector3 velocity;
    public float mass;
    public Planet planeta;

    private Vector3 gravityForce;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        velocity = Vector3.right * speed;
    }

    // Update is called once per frame
    void Update()
    {
        if (planeta != null)
        {
            gravityForce = planeta.GetGravityForce();
            Debug.Log("Planeta detectado");
        }

        velocity += gravityForce * Time.deltaTime;
        transform.position += velocity * Time.deltaTime;


    }
}
