using System.Runtime.InteropServices;
using UnityEngine;

public class GravityGPUManager : MonoBehaviour
{[Header("Assets")]
    public ComputeShader planetsCS; // PlanetsInit.compute
    public ComputeShader shipsCS;   // ShipsSim.compute
    public GameObject planetPrefab;
    public GameObject shipPrefab;

    [Header("Counts")]
    public int planetCount = 8;
    public int shipCount = 2000;

    [Header("Spawn Area")]
    public Vector3 areaMin = new Vector3(-30, -15, -30);
    public Vector3 areaMax = new Vector3(30, 15, 30);

    [Header("Planets")]
    public Vector2 radiusRange = new Vector2(1f, 4f);
    public float massPerRadius = 100f;

    [Header("Ships")]
    public Vector2 speedRange = new Vector2(3, 12);

    [Header("Physics")]
    public float G = 10f;
    public bool bounce = true;
    public float bounceDamping = 0.9f;
    public float minDistance = 0.25f;

    [Header("Random")]
    public uint seed = 12345;

    // Compute buffers
    private ComputeBuffer planetBuffer;
    private ComputeBuffer shipBuffer;

    // CPU arrays espejo
    [StructLayout(LayoutKind.Sequential)]
    struct PlanetData
    {
        public Vector4 posRad;   // xyz = position, w = radius
        public Vector2 mass_pad; // x = mass, y = pad (padding)
    }

    [StructLayout(LayoutKind.Sequential)]
    struct ShipData
    {
        public Vector4 pos; // xyz = position, w = padding
        public Vector4 vel; // xyz = velocity, w = padding
    }

    private PlanetData[] planetsCPU;
    private ShipData[] shipsCPU;

    private Transform[] planetTransforms;
    private Transform[] shipTransforms;

    // Kernels
    private int kInitPlanets, kInitShips, kUpdateShips;
    private uint tgxInitPlanets = 64, tgxInitShips = 64, tgxUpdateShips = 64;

    void Start()
    {
        // Validar asignación de shaders
        if (!planetsCS || !shipsCS)
        {
            Debug.LogError("Asigná ambos ComputeShaders en el inspector.");
            enabled = false;
            return;
        }

        // Obtener kernels válidos
        kInitPlanets = planetsCS.FindKernel("InitPlanets");
        kInitShips = shipsCS.FindKernel("InitShips");
        kUpdateShips = shipsCS.FindKernel("UpdateShips");

        if (kInitPlanets < 0 || kInitShips < 0 || kUpdateShips < 0)
        {
            Debug.LogError($"Error en FindKernel: InitPlanets={kInitPlanets}, InitShips={kInitShips}, UpdateShips={kUpdateShips}");
            enabled = false;
            return;
        }

        // Obtener tamaños de grupo de hilos
        planetsCS.GetKernelThreadGroupSizes(kInitPlanets, out tgxInitPlanets, out _, out _);
        shipsCS.GetKernelThreadGroupSizes(kInitShips, out tgxInitShips, out _, out _);
        shipsCS.GetKernelThreadGroupSizes(kUpdateShips, out tgxUpdateShips, out _, out _);

        // Crear arrays espejo
        planetsCPU = new PlanetData[planetCount];
        shipsCPU = new ShipData[shipCount];

        // Crear buffers de GPU
        planetBuffer = new ComputeBuffer(Mathf.Max(1, planetCount), Marshal.SizeOf(typeof(PlanetData)));
        shipBuffer = new ComputeBuffer(Mathf.Max(1, shipCount), Marshal.SizeOf(typeof(ShipData)));

        // Parámetros y buffers en planetsCS (planetas)
        planetsCS.SetInt("_PlanetCount", planetCount);
        planetsCS.SetVector("_AreaMin", areaMin);
        planetsCS.SetVector("_AreaMax", areaMax);
        planetsCS.SetVector("_RadiusRange", radiusRange);
        planetsCS.SetFloat("_MassPerRadius", massPerRadius);
        planetsCS.SetInt("_Seed", (int)seed);

        planetsCS.SetBuffer(kInitPlanets, "_Planets", planetBuffer);

        shipsCS.SetInt("_PlanetCount", planetCount);
        shipsCS.SetInt("_ShipCount", shipCount);
        shipsCS.SetVector("_AreaMin", areaMin);
        shipsCS.SetVector("_AreaMax", areaMax);
        shipsCS.SetFloat("_G", G);
        shipsCS.SetVector("_ShipSpeedRange", speedRange);
        shipsCS.SetFloat("_Bounce", bounce ? 1f : 0f);
        shipsCS.SetFloat("_Damping", bounceDamping);
        shipsCS.SetFloat("_MinDist", minDistance);
        shipsCS.SetInt("_Seed", (int)seed);

        shipsCS.SetBuffer(kInitShips, "_Ships", shipBuffer);
        shipsCS.SetBuffer(kInitShips, "_Planets", planetBuffer);
        shipsCS.SetBuffer(kUpdateShips, "_Ships", shipBuffer);
        shipsCS.SetBuffer(kUpdateShips, "_Planets", planetBuffer);

        // Inicializar planetas y naves en GPU
        int groupsPlanets = Mathf.Max(1, Mathf.CeilToInt(planetCount / (float)tgxInitPlanets));
        int groupsShips = Mathf.Max(1, Mathf.CeilToInt(shipCount / (float)tgxInitShips));
        planetsCS.Dispatch(kInitPlanets, groupsPlanets, 1, 1);
        shipsCS.Dispatch(kInitShips, groupsShips, 1, 1);

        // Leer datos inicializados
        planetBuffer.GetData(planetsCPU);
        shipBuffer.GetData(shipsCPU);

        // Instanciar Planetas en la escena
        planetTransforms = new Transform[planetCount];
        for (int i = 0; i < planetCount; ++i)
        {
            GameObject go = Instantiate(planetPrefab);
            go.transform.position = new Vector3(planetsCPU[i].posRad.x, planetsCPU[i].posRad.y, planetsCPU[i].posRad.z);
            float scale = planetsCPU[i].posRad.w * 2f; // Diametro = 2 * radio
            go.transform.localScale = Vector3.one * scale;
            planetTransforms[i] = go.transform;
        }

        // Instanciar Naves en la escena
        shipTransforms = new Transform[shipCount];
        for (int i = 0; i < shipCount; ++i)
        {
            GameObject go = Instantiate(shipPrefab);
            go.transform.position = new Vector3(shipsCPU[i].pos.x, shipsCPU[i].pos.y, shipsCPU[i].pos.z);
            shipTransforms[i] = go.transform;
        }
    }

    void Update()
    {
        if (shipTransforms == null) return;  // Proteccion contra NullReference si Start falló

        shipsCS.SetFloat("_DeltaTime", Time.deltaTime);

        // Rebindeo por kernel (especialmente importante tras recompilar y play/reload)
        shipsCS.SetBuffer(kUpdateShips, "_Ships", shipBuffer);
        shipsCS.SetBuffer(kUpdateShips, "_Planets", planetBuffer);

        int groups = Mathf.Max(1, Mathf.CeilToInt(shipCount / (float)tgxUpdateShips));
        shipsCS.Dispatch(kUpdateShips, groups, 1, 1);

        // Leer posiciones de vuelta
        shipBuffer.GetData(shipsCPU);

        // Actualizar posiciones en escena
        for (int i = 0; i < shipCount; i++)
            shipTransforms[i].position = new Vector3(shipsCPU[i].pos.x, shipsCPU[i].pos.y, shipsCPU[i].pos.z);
    }

    void OnDestroy()
    {
        if (planetBuffer != null) planetBuffer.Dispose();
        if (shipBuffer != null) shipBuffer.Dispose();
    }

}