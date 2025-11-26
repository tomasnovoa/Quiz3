using System.Runtime.InteropServices;
using UnityEngine;

public class GravityGPUManager : MonoBehaviour
{
    
    public ComputeShader planetsCS;          
    public ComputeShader shipsCS;           
    public GameObject planetPrefab;
    public GameObject shipPrefab;

    [Header("Counts")]
    public int planetCount = 8;
    public int shipCount = 2000;

    [Header("Area")]
    public Vector3 areaMin = new Vector3(-30, -15, -30);
    public Vector3 areaMax = new Vector3(30, 15, 30);

    [Header("Planetas")]
    public Vector2 radiusRange = new Vector2(1f, 4f);
    public float massPerRadius = 100f;

    [Header("Naves")]
    public Vector2 speedRange = new Vector2(3f, 12f);

    [Header("Fisicas")]
    public float G = 10f;
    public bool bounce = true;
    public float bounceDamping = 0.9f;
    public float minDistance = 0.25f;

    [Header("Random")]
    public uint seed = 12345;

    [Header("Grilla")]
    public float gridWidthWorld = 16f;  
    public float gridHeightWorld = 9f;   
    public float borderPadding = 0.5f;
    public float uniformRadius = 0.2f;   

    
    private ComputeBuffer planetBuffer;
    private ComputeBuffer shipBuffer;

    // Estructuras de los cs

    [StructLayout(LayoutKind.Sequential)]
    private struct PlanetData
    {
        public Vector4 posRad;   // xyz = position, w = radius
        public Vector2 mass_pad; // x = mass, y = padding
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ShipData
    {
        public Vector4 pos;      // xyz = position
        public Vector4 vel;      // xyz = velocity
    }

    private PlanetData[] planetsCPU;
    private ShipData[] shipsCPU;

    private Transform[] planetTransforms;
    private Transform[] shipTransforms;

    // Kernels y tamaños de grupo 

    private int kInitPlanets;
    private int kInitShips;
    private int kUpdateShips;

    private uint tgxInitPlanets = 64;
    private uint tgxInitShips = 64;
    private uint tgxUpdateShips = 64;


    private void Start()
    {
        if (!ValidateShaders()) return;

        InicializarKernels();
        InicializarBuffersAndArrays();
        SeteoParametrosShaders();
        InicializarGPUData();
        InicializarGrilla();
    }

    private void Update()
    {
        if (shipTransforms == null) return;

        UpdateShipsGPU();
        ApplyShipPositionsToScene();
    }

    private void OnDestroy()
    {
        planetBuffer?.Dispose();
        shipBuffer?.Dispose();
    }

   

    private bool ValidateShaders()
    {
        if (!planetsCS || !shipsCS)
        {
            Debug.LogError("Asigna planetsCS y shipsCS en el inspector.");
            enabled = false;
            return false;
        }

        return true;
    }

    private void InicializarKernels()
    {
        kInitPlanets = planetsCS.FindKernel("InitPlanets");
        kInitShips = shipsCS.FindKernel("InitShips");
        kUpdateShips = shipsCS.FindKernel("UpdateShips");

        if (kInitPlanets < 0 || kInitShips < 0 || kUpdateShips < 0)
        {
            Debug.LogError(
                $"FindKernel fallo: InitPlanets={kInitPlanets}, " +
                $"InitShips={kInitShips}, UpdateShips={kUpdateShips}"
            );
            enabled = false;
            return;
        }

        planetsCS.GetKernelThreadGroupSizes(kInitPlanets, out tgxInitPlanets, out _, out _);
        shipsCS.GetKernelThreadGroupSizes(kInitShips, out tgxInitShips, out _, out _);
        shipsCS.GetKernelThreadGroupSizes(kUpdateShips, out tgxUpdateShips, out _, out _);
    }

    private void InicializarBuffersAndArrays()
    {
        planetsCPU = new PlanetData[planetCount];
        shipsCPU = new ShipData[shipCount];

        planetBuffer = new ComputeBuffer(
            Mathf.Max(1, planetCount),
            Marshal.SizeOf(typeof(PlanetData))
        );

        shipBuffer = new ComputeBuffer(
            Mathf.Max(1, shipCount),
            Marshal.SizeOf(typeof(ShipData))
        );
    }

    private void SeteoParametrosShaders()
    {
        // Planeta shader
        planetsCS.SetInt("_PlanetCount", planetCount);
        planetsCS.SetVector("_AreaMin", areaMin);
        planetsCS.SetVector("_AreaMax", areaMax);
        planetsCS.SetVector("_RadiusRange", new Vector2(uniformRadius, uniformRadius));
        planetsCS.SetFloat("_MassPerRadius", massPerRadius);
        planetsCS.SetInt("_Seed", (int)seed);
        planetsCS.SetBuffer(kInitPlanets, "_Planets", planetBuffer);

        // Naves shader
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
    }

    private void InicializarGPUData()
    {
        int groupsP = Mathf.Max(1, Mathf.CeilToInt(planetCount / (float)tgxInitPlanets));
        int groupsS = Mathf.Max(1, Mathf.CeilToInt(shipCount / (float)tgxInitShips));

        planetsCS.Dispatch(kInitPlanets, groupsP, 1, 1);
        shipsCS.Dispatch(kInitShips, groupsS, 1, 1);

        planetBuffer.GetData(planetsCPU);
        shipBuffer.GetData(shipsCPU);
    }



    private void InicializarGrilla()
    {
        // Cálculo de grilla 2D para que todo quepa en la cámara
        int totalCells = Mathf.Max(planetCount, shipCount);
        int cols = Mathf.CeilToInt(Mathf.Sqrt(totalCells));
        int rows = Mathf.CeilToInt(totalCells / (float)cols);

        float usableW = gridWidthWorld - 2f * borderPadding;
        float usableH = gridHeightWorld - 2f * borderPadding;

        float cellW = usableW / cols;
        float cellH = usableH / rows;
        float cellSize = Mathf.Min(cellW, cellH);

        float originX = -usableW * 0.5f;
        float originY = -usableH * 0.5f;

        float visualDiameter = cellSize * 0.7f;          // 70% del tamaño de la celda
        Vector3 uniformScale = Vector3.one * visualDiameter;

        // Instanciar planetas en grilla
        planetTransforms = new Transform[planetCount];

        for (int i = 0; i < planetCount; ++i)
        {
            int cx = i % cols;
            int cy = i / cols;

            float x = originX + (cx + 0.5f) * cellSize;
            float y = originY + (cy + 0.5f) * cellSize;

            GameObject go = Instantiate(planetPrefab);
            go.transform.position = new Vector3(x, y, 0f);
            go.transform.localScale = uniformScale;
            planetTransforms[i] = go.transform;

            planetsCPU[i].posRad = new Vector4(x, y, 0f, uniformRadius);
        }

        // Instanciar naves en la misma grilla
        shipTransforms = new Transform[shipCount];

        for (int i = 0; i < shipCount; ++i)
        {
            int index = i; // si quieres que no compartan celdas: i + planetCount
            int cx = index % cols;
            int cy = index / cols;

            float x = originX + (cx + 0.5f) * cellSize;
            float y = originY + (cy + 0.5f) * cellSize;

            GameObject go = Instantiate(shipPrefab);
            go.transform.position = new Vector3(x, y, 0f);
            go.transform.localScale = uniformScale;
            shipTransforms[i] = go.transform;

            shipsCPU[i].pos = new Vector4(x, y, 0f, 0f);
        }

        // Subir posiciones de grilla a GPU para que la sim arranque desde ahí
        planetBuffer.SetData(planetsCPU);
        shipBuffer.SetData(shipsCPU);
    }

    

    // Actualización de la simulación
    private void UpdateShipsGPU()
    {
        shipsCS.SetFloat("_DeltaTime", Time.deltaTime);

        // enlazar recursos por kernel
        shipsCS.SetBuffer(kUpdateShips, "_Ships", shipBuffer);
        shipsCS.SetBuffer(kUpdateShips, "_Planets", planetBuffer);

        int groups = Mathf.Max(1, Mathf.CeilToInt(shipCount / (float)tgxUpdateShips));
        shipsCS.Dispatch(kUpdateShips, groups, 1, 1);

        shipBuffer.GetData(shipsCPU);
    }

    private void ApplyShipPositionsToScene()
    {
        for (int i = 0; i < shipCount; i++)
        {
            Vector4 p = shipsCPU[i].pos;
            shipTransforms[i].position = new Vector3(p.x, p.y, p.z);
        }
    }
}
