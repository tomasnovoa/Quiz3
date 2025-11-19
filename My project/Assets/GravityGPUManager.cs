using System.Runtime.InteropServices;
using UnityEngine;

public class GravityGPUManager : MonoBehaviour
{
    [Header("Assets")]
    public ComputeShader planetsCS; // PlanetsInit.compute
    public ComputeShader shipsCS;   // ShipsSim.compute
    public GameObject planetPrefab;
    public GameObject shipPrefab;

    [Header("Counts")]
    public int planetCount = 8;
    public int shipCount = 2000;

    [Header("Spawn Area")]
    public Vector3 areaMin = new(-30, -15, -30);
    public Vector3 areaMax = new(30, 15, 30);

    [Header("Planets")]
    public Vector2 radiusRange = new(1f, 4f);
    public float massPerRadius = 100f;

    [Header("Ships")]
    public Vector2 speedRange = new(3, 12);

    [Header("Physics")]
    public float G = 10f;
    public bool bounce = true;
    public float bounceDamping = 0.9f;
    public float minDistance = 0.25f;

    [Header("Random")]
    public uint seed = 12345;

    ComputeBuffer planetBuffer, shipBuffer;

    [StructLayout(LayoutKind.Sequential)]
    struct PlanetData { public Vector4 posRad; public Vector2 mass_pad; }
    [StructLayout(LayoutKind.Sequential)]
    struct ShipData { public Vector4 pos; public Vector4 vel; }

    int kInitPlanets, kInitShips, kUpdateShips;
    uint tgxInitPlanets = 64, tgxInitShips = 64, tgxUpdateShips = 64;

    Transform[] planetsTf, shipsTf;
    PlanetData[] planetsCPU;
    ShipData[] shipsCPU;

    void Start()
    {
        // después de asignar planetsCS y shipsCS en el Inspector:
        kInitPlanets = planetsCS.FindKernel("InitPlanets");
        kInitShips = shipsCS.FindKernel("InitShips");
        kUpdateShips = shipsCS.FindKernel("UpdateShips");

        if (planetsCS == null || shipsCS == null)
        { Debug.LogError("Asigna ambos ComputeShaders en el Inspector"); enabled = false; return; } // evita NullRef [web:22]

        if (kInitPlanets < 0 || kInitShips < 0 || kUpdateShips < 0)
        { Debug.LogError($"FindKernel fallo: P:{kInitPlanets} S:{kInitShips} U:{kUpdateShips}"); enabled = false; return; } // [web:22]

        // tamaños de grupo seguros
        uint sxP, syP, szP, sxS, syS, szS, sxU, syU, szU;
        planetsCS.GetKernelThreadGroupSizes(kInitPlanets, out sxP, out syP, out szP);   // usa handle válido [web:50]
        shipsCS.GetKernelThreadGroupSizes(kInitShips, out sxS, out syS, out szS);     // [web:50]
        shipsCS.GetKernelThreadGroupSizes(kUpdateShips, out sxU, out syU, out szU);     // [web:50]

        // buffers por kernel (en Start, antes del primer Dispatch)
        planetsCS.SetBuffer(kInitPlanets, "_Planets", planetBuffer);                    // escribe planetas [web:46]
        shipsCS.SetBuffer(kInitShips, "_Ships", shipBuffer);                         // escribe naves [web:46]
        shipsCS.SetBuffer(kInitShips, "_Planets", planetBuffer);                       // lee planetas [web:46]
        shipsCS.SetBuffer(kUpdateShips, "_Ships", shipBuffer);                         // [web:46]
        shipsCS.SetBuffer(kUpdateShips, "_Planets", planetBuffer);                       // [web:46]

        // despachos (usa ceil para grupos >=1)
        int gpP = Mathf.Max(1, Mathf.CeilToInt(planetCount / (float)sxP));
        int gpS = Mathf.Max(1, Mathf.CeilToInt(shipCount / (float)sxS));
        int gpU = Mathf.Max(1, Mathf.CeilToInt(shipCount / (float)sxU));
        planetsCS.Dispatch(kInitPlanets, gpP, 1, 1);                                     // [web:24]
        shipsCS.Dispatch(kInitShips, gpS, 1, 1);                                     // [web:24]

        // en Update, re‑bindea por si hubo reimport/reload y despacha
        shipsCS.SetBuffer(kUpdateShips, "_Ships", shipBuffer);                          // binding por kernel [web:46]
        shipsCS.SetBuffer(kUpdateShips, "_Planets", planetBuffer);                        // binding por kernel [web:46]
        shipsCS.SetFloat("_DeltaTime", Time.deltaTime);
        shipsCS.Dispatch(kUpdateShips, gpU, 1, 1);                                       // [web:24]

    }

}