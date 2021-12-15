using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ElasticFemImplicit2D : MonoBehaviour
{
    Mesh mesh;
    MeshFilter meshFilter;
    Vector3[] VerteicesArray = new Vector3[node_num];
    Vector2[] uvs = new Vector2[node_num];

    const int node_row_num = 12;
    const float mass = 100f;
    float dt = 2.0f;
    bool enableCollsion = true;
    bool enableGravity = true;
    bool enableRightStretch = false;
    bool enableRightTopStretch = true;



    const int node_num = node_row_num * node_row_num;
    const int element_row_num = node_row_num - 1;
    const int element_num = element_row_num * element_row_num * 2;

    Vector2[] node_pos = new Vector2[node_num];
    Vector2[] node_vel = new Vector2[node_num];
    Vector2[] node_force = new Vector2[node_num];
    Vector2[] node_pos_correct = new Vector2[node_num];

    Matrix2x2[] element_Dm_inv = new Matrix2x2[element_num];
    int[] element_idx = new int[element_num * 3];
    float[] element_area = new float[element_num];

    float lame_mu = 8f;
    float lame_la = 8f;

    float dx_space = 8.0f / node_row_num;// /  node_row_num;

    Matrix2x2[] dD = new Matrix2x2[6]; // 位置求导
    Matrix2x2[] dF = new Matrix2x2[6]; // 变形梯度求导
    Matrix2x2[] dE = new Matrix2x2[6]; // 应变求导
    Matrix2x2[] dP = new Matrix2x2[6]; // 第一piola 应力求导
    Matrix2x2[] dH = new Matrix2x2[6]; // hessian 求导



    const int K_row = node_num * 2;
    float[] Kmat = new float[K_row * K_row];
    float[] Amat = new float[K_row * K_row];
    float[] bvec = new float[K_row];
    float[] xvec = new float[K_row];

    GameObject[] sphere;
    Vector2[] spherePosition;
    float[] sphereRadius;

    public Camera camera;
    private Ray ra;
    private RaycastHit hit;
    void Start()
    {
        meshFilter = GetComponent<MeshFilter>();
        mesh = new Mesh();
        meshFilter.mesh = mesh;
        sphere = GameObject.FindGameObjectsWithTag("SphereObstacle");
        spherePosition = new Vector2[sphere.Length];
        sphereRadius = new float[sphere.Length];
        for (int i = 0; i < sphere.Length; i++)
        {
            spherePosition[i] = new Vector2(sphere[i].transform.position.x, sphere[i].transform.position.y);
            sphereRadius[i] = sphere[i].transform.localScale.x * 0.5f;
        }
        drawTriangle();
        InitFem();
        UpdatePos();
        camera = GameObject.FindGameObjectWithTag("MainCamera").GetComponent<Camera>();
    }
    void drawTriangle()
    {
        for (int j = 0; j < node_row_num; j++)
        {
            for (int i = 0; i < node_row_num; i++)
            {
                int idx = j * node_row_num + i;
                VerteicesArray[idx] = new Vector3(i, j, 0) * dx_space;
                node_pos[idx] = new Vector2(i, j) * dx_space;

                node_vel[idx] = new Vector2(0, 0);
                node_force[idx] = new Vector2(0, 0);
                node_pos_correct[idx] = new Vector2(0, 0);

                uvs[idx] = new Vector2(0.9f - (float)i / node_row_num, (float)j / node_row_num);
            }
        }

        int cnt = 0;
        for (int j = 0; j < element_row_num; j++)
        {
            for (int i = 0; i < element_row_num; i++)
            {
                element_idx[cnt++] = j * node_row_num + i;
                element_idx[cnt++] = j * node_row_num + i + 1;
                element_idx[cnt++] = j * node_row_num + i + node_row_num;

                element_idx[cnt++] = j * node_row_num + i + 1;
                element_idx[cnt++] = j * node_row_num + i + 1 + node_row_num;
                element_idx[cnt++] = j * node_row_num + i + node_row_num;
            }
        }
        //add these two triangles to the mesh
        mesh.vertices = VerteicesArray;
        mesh.triangles = element_idx;
        mesh.uv = uvs;
    }

    void InitFem()
    {
        for (int ie = 0; ie < element_num; ie++)
        {
            Vector2 pos0 = node_pos[element_idx[ie * 3 + 0]];
            Vector2 pos1 = node_pos[element_idx[ie * 3 + 1]];
            Vector2 pos2 = node_pos[element_idx[ie * 3 + 2]];
            float dX_00 = pos1.x - pos0.x;
            float dX_01 = pos2.x - pos0.x;
            float dX_10 = pos1.y - pos0.y;
            float dX_11 = pos2.y - pos0.y;


            float det = (dX_00 * dX_11 - dX_01 * dX_10);
            float det_inv = 1.0f / det;
            element_area[ie] = det * 0.5f;
            element_Dm_inv[ie] = new Matrix2x2(dX_11 * det_inv, -dX_01 * det_inv, -dX_10 * det_inv, dX_00 * det_inv);

        }
        if (enableRightTopStretch)
            node_pos[node_row_num - 1] += new Vector2(1.0f, 0) * dx_space;
        if (enableRightStretch)
            node_pos[1] = new Vector2(2, 0);
        for (int i = 0; i < K_row * K_row; i++)
        {
            Kmat[i] = Amat[i] = 0;
        }


    }
    void Step()
    {
        for (int ie = 0; ie < element_num; ie++)
        {
            Vector2 pos0 = node_pos[element_idx[ie * 3 + 0]];
            Vector2 pos1 = node_pos[element_idx[ie * 3 + 1]];
            Vector2 pos2 = node_pos[element_idx[ie * 3 + 2]];

            Matrix2x2 dx = new Matrix2x2(pos1.x - pos0.x, pos2.x - pos0.x, pos1.y - pos0.y, pos2.y - pos0.y);

            dx.debug("dx", ie);
            Matrix2x2 F = Matrix2x2.mul(dx, element_Dm_inv[ie]);
            F.debug("F", ie);
            Matrix2x2 E = (Matrix2x2.mul(F.transpose(), F) - Matrix2x2.identity()) * 0.5f;
            E.debug("E", ie);
            Matrix2x2 piola = Matrix2x2.mul(F, 2 * lame_mu * E + lame_la * (E.v00 + E.v11) * Matrix2x2.identity());
            piola.debug("piola", ie);
            Matrix2x2 H = -element_area[ie] * Matrix2x2.mul(piola, element_Dm_inv[ie].transpose());

            Vector2 gradC1 = new Vector2(H.v00, H.v10);
            Vector2 gradC2 = new Vector2(H.v01, H.v11);
            Vector2 gradC0 = -gradC1 - gradC2;

            node_force[element_idx[ie * 3 + 0]] += gradC0;
            node_force[element_idx[ie * 3 + 1]] += gradC1;
            node_force[element_idx[ie * 3 + 2]] += gradC2;

            dD[0] = new Matrix2x2(-1, -1, 0, 0);
            dD[1] = new Matrix2x2(0, 0, -1, -1);
            dD[2] = new Matrix2x2(1, 0, 0, 0);
            dD[3] = new Matrix2x2(0, 0, 1, 0);
            dD[4] = new Matrix2x2(0, 1, 0, 0);
            dD[5] = new Matrix2x2(0, 0, 0, 1);
            element_Dm_inv[ie].debug("element_inv" + ie, ie);
            for (int nd = 0; nd < 6; nd++)
            {
                dF[nd] = Matrix2x2.mul(dD[nd], element_Dm_inv[ie]);
                dF[nd].debug("dF[" + nd + "]", ie);
                dE[nd] = (Matrix2x2.mul(dF[nd].transpose(), F) + Matrix2x2.mul(F.transpose(), dF[nd])) * 0.5f;
                dE[nd].debug("dE[" + nd + "]", ie);
                dP[nd] = Matrix2x2.mul(dF[nd], 2 * lame_mu * E + lame_la * (E.v00 + E.v11) * Matrix2x2.identity());
                dP[nd] += Matrix2x2.mul(F, 2 * lame_mu * dE[nd] + lame_la * (dE[nd].v00 + dE[nd].v11) * Matrix2x2.identity());
                dP[nd].debug("dP[" + nd + "]", ie);
                dH[nd] = -element_area[ie] * Matrix2x2.mul(dP[nd], element_Dm_inv[ie].transpose());
                dH[nd].debug("dH[" + nd + "]", ie);
            }
            for (int n = 0; n < 3; n++)
            {
                int nidx = element_idx[ie * 3 + n];
                for (int d = 0; d < 2; d++)
                {
                    int kidx = nidx * 2 + d;
                    int didx = n * 2 + d;
                    int idx;
                    idx = element_idx[ie * 3 + 1] * 2;
                    Kmat[idx * K_row + kidx] += dH[didx].v00;
                    idx = element_idx[ie * 3 + 1] * 2 + 1;
                    Kmat[idx * K_row + kidx] += dH[didx].v10;
                    idx = element_idx[ie * 3 + 2] * 2;
                    Kmat[idx * K_row + kidx] += dH[didx].v01;
                    idx = element_idx[ie * 3 + 2] * 2 + 1;
                    Kmat[idx * K_row + kidx] += dH[didx].v11;
                    idx = element_idx[ie * 3 + 0] * 2;
                    Kmat[idx * K_row + kidx] += -dH[didx].v00 - dH[didx].v01;
                    idx = element_idx[ie * 3 + 0] * 2 + 1;
                    Kmat[idx * K_row + kidx] += -dH[didx].v10 - dH[didx].v11;
                }
            }
        }
    }

    void AssemblyAndSolve()
    {
        for (int j = 0; j < node_row_num; j++)
        {
            for (int i = 0; i < node_row_num; i++)
            {
                int idx = j * node_row_num + i;
                for (int k = 0; k < sphere.Length; k++)
                {
                    float sqrDis = Vector2.SqrMagnitude(node_pos[idx] + node_pos_correct[idx] - spherePosition[k]);
                    if (sqrDis < sphereRadius[k] * sphereRadius[k])
                    {
                        if (enableCollsion)
                        {
                            //node_force[idx] += (node_pos[idx] + node_pos_correct[idx] - spherePosition[k]).normalized * (sphereRadius[k] * sphereRadius[k] - sqrDis) * 1.0f;
                            node_force[idx] += (node_pos[idx] + node_pos_correct[idx] - spherePosition[k]).normalized  * 0.2f;
                        }
                            
                    }
                    else
                    {
                        if (enableGravity)
                            node_force[idx] += new Vector2(0, -0.001f);
                    }
                }
            }
        }

        for (int j = 0; j < K_row; j++)
        {
            for (int i = 0; i < K_row; i++)
            {
                int idx = j * K_row + i;
                if (i == j)
                {
                    Amat[idx] = mass - Kmat[idx] * dt * dt;
                }
                else
                {
                    Amat[idx] = -Kmat[idx] * dt * dt;
                }
                //Debug.Log("Kmat[" + j + "," + i + "]" + Kmat[idx]);
                Kmat[idx] = 0;
            }
        }

        for (int i = 0; i < node_num; i++)
        {
            int idx = i / 2;
            xvec[i * 2 + 0] = node_vel[i].x;
            xvec[i * 2 + 1] = node_vel[i].y;
            bvec[i * 2 + 0] = node_vel[i].x * mass + dt * node_force[i].x;
            bvec[i * 2 + 1] = node_vel[i].y * mass + dt * node_force[i].y;
        }
        cgSolver solver = new cgSolver();
        xvec = solver.DenseSolver(K_row, Amat, xvec, bvec);
        for (int i = 0; i < node_num; i++)
        {
            int idx = i / 2;
            float damp = 0.6f;
            node_pos_correct[i].x = (xvec[i * 2 + 0] * (1 - damp) + damp * node_vel[i].x) *dt;
            node_pos_correct[i].y = (xvec[i * 2 + 1] * (1 - damp) + damp * node_vel[i].y) * dt;
           // node_pos_correct[i].x = xvec[i * 2 + 0] * dt;
            //node_pos_correct[i].y = xvec[i * 2 + 1] * dt;
            node_vel[i].x = xvec[i * 2 + 0];
            node_vel[i].y = xvec[i * 2 + 1];
        }
        for (int i = 0; i < K_row; i++)
        {
            // Debug.Log("xvec " + i + " = " + xvec[i]);
        }
    }
    void UpdatePos()
    {
        for (int j = 0; j < node_row_num; j++)
        {
            for (int i = 0; i < node_row_num; i++)
            {

                int idx = j * node_row_num + i;
                node_pos[idx] += node_pos_correct[idx];
                VerteicesArray[idx] = new Vector3(node_pos[idx].x, node_pos[idx].y, 0);
                node_force[idx] = new Vector2(0, 0);
                node_pos_correct[idx] = new Vector2(0, 0);
            }
        }
        mesh.vertices = VerteicesArray;
        mesh.RecalculateNormals();
    }
    // Update is called once per frame
    bool NeedFindNewnode = true;
    int idx = -1;

    void Pick()
    {
        Vector3 mousePos = Input.mousePosition;
        mousePos.z = Camera.main.transform.position.z;
        Vector3 worldPosition = Camera.main.ScreenToWorldPoint(mousePos);
        Vector2 mouse_pos = new Vector2(worldPosition.x, worldPosition.y);
        if (Input.GetMouseButton(0))
        {
                float maxDis = 10f;
                for(int i = 0; i < node_num;i++)
                {
                    float dis = Vector2.SqrMagnitude(node_pos[i] - mouse_pos);
                    if (dis < maxDis)
                    {
                        Vector2 dir = mouse_pos - node_pos[i];
                    node_force[i] += dir.normalized * 1.0f;
                    }
                
                }
        }
    }
    void Update()
    {

        Step();
        Pick();
        AssemblyAndSolve();
        UpdatePos();
    }
}
