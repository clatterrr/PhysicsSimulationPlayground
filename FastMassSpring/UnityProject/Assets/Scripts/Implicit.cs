using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Implicit : MonoBehaviour
{
    const int node_row_num = 16;
    const int node_num = node_row_num * node_row_num;
    const int element_row_num = node_row_num - 1;
    const int element_num = element_row_num * element_row_num * 2;
    int[] element_idx = new int[element_num * 3];

    Mesh mesh;
    MeshFilter meshFilter;
    public GameObject sphereCollision;
    Vector3 spherePos;
    float sphereRadius;

    Vector3[] node_pos = new Vector3[node_num];
    Vector3[] node_force = new Vector3[node_num];
    float[] node_mass = new float[node_num];
    Vector2[] uvs = new Vector2[node_num];

    const int stretch_num = (node_row_num - 1) * node_row_num * 2;
    const int shear_num = (node_row_num - 1) * (node_row_num - 1) * 2;
    const int bend_num = (node_row_num - 2) * node_row_num * 2;
    const int constraint_num = stretch_num + shear_num + bend_num;

    float[] stiffness = new float[constraint_num];

    const int row = node_num * 3;
    

    int[] d_st = new int[constraint_num];
    int[] d_ed = new int[constraint_num];
    float[] rest = new float[constraint_num];
    float[] f = new float[row];
    float[] df = new float[row * row];
    float[] A = new float[row * row];
    float[] inertia = new float[row];
    float[] pos = new float[row];
    float[] pos_pre = new float[row];
    float[] rhs = new float[row];
    float[] dfdx_x = new float[row];
    float[] iden3 = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    float dt = 1f;

    int cnt = 0;

    cgSolver solver = new cgSolver();
    void Start()
    {
        meshFilter = GetComponent<MeshFilter>();
        mesh = new Mesh();
        meshFilter.mesh = mesh;
        drawTriangle();
        InitConstraint();
        pos[0] = -1;
        spherePos = sphereCollision.transform.position;
        sphereRadius = sphereCollision.transform.localScale.x * 0.6f;
        for (int i = 0; i < row; i++) pos_pre[i] = pos[i];
    }

    void drawTriangle()
    {
        float dx_space = 10.0f / (node_row_num - 1);
        for (int j = 0; j < node_row_num; j++)
        {
            for (int i = 0; i < node_row_num; i++)
            {
                int idx = j * node_row_num + i;
                node_pos[idx] = new Vector3(i, 5,j) * dx_space;
                pos[idx * 3 + 0] = pos_pre[idx * 3 + 0] = node_pos[idx].x;
                pos[idx * 3 + 1] = pos_pre[idx * 3 + 1] = node_pos[idx].y;
                pos[idx * 3 + 2] = pos_pre[idx * 3 + 2] = node_pos[idx].z;
                node_force[idx] = Vector3.zero;
                node_mass[idx] = 1;
                uvs[idx] = new Vector2((float)i / (node_row_num - 1), (float)j / (node_row_num - 1));
            }
        }

        int cnt = 0;
        for (int j = 0; j < element_row_num; j++)
        {
            for (int i = 0; i < element_row_num; i++)
            {
                element_idx[cnt++] = j * node_row_num + i;
                element_idx[cnt++] = j * node_row_num + i + node_row_num;
                element_idx[cnt++] = j * node_row_num + i + 1;

                element_idx[cnt++] = j * node_row_num + i + 1;
                element_idx[cnt++] = j * node_row_num + i + node_row_num;
                element_idx[cnt++] = j * node_row_num + i + 1 + node_row_num;
            }
        }
        //add these two triangles to the mesh
        mesh.vertices = node_pos;
        mesh.triangles = element_idx;
        mesh.uv = uvs;
    }
    void ConstructMatrix(int st, int ed)
    {
        d_st[cnt] = st;
        d_ed[cnt] = ed;
        rest[cnt] = Vector3.Distance(node_pos[st], node_pos[ed]);
        stiffness[cnt] = 1;
        cnt += 1;
    }
    void InitConstraint()
    {

        for (int j = 0; j < node_row_num; j++)
        {
            for (int i = 0; i < node_row_num; i++)
            {
                int idx = j * node_row_num + i;
                // stretch
                if (i < node_row_num - 1) ConstructMatrix(idx, idx + 1);
                if (j < node_row_num - 1) ConstructMatrix(idx, idx + node_row_num);
                // bend
                if (i < node_row_num - 2) ConstructMatrix(idx, idx + 2);
                if (j < node_row_num - 2) ConstructMatrix(idx, idx + node_row_num * 2);
                // shear
                if ((i < node_row_num - 1) && (j < node_row_num - 1))
                {
                    ConstructMatrix(idx, idx + 1 + node_row_num);
                    ConstructMatrix(idx + 1, idx + node_row_num);
                }
            }
        }
        for (int i = 0; i < row * row; i++) df[i] = 0;
        for (int i = 0; i < row; i++) f[i] = 0;
    }
    float[] MatrixMultipleVec(int size1, int size2, float[] Mat, float[] vec)
    {
        float[] res = new float[size1];
        for (int i = 0; i < size1; i++)
        {
            res[i] = 0;
            for (int j = 0; j < size2; j++)
            {
                res[i] += Mat[i * size2 + j] * vec[j];
            }
        }

        return res;
    }
    void DebugVec(int row, float[] vec, string name)
    {
        for (int i = 0; i < row; i++)
        {
            Debug.Log("vec" + name + " i = " + i + " val = " + vec[i]);
        }
    }
    void DebugVec(int row, int[] vec, string name)
    {
        for (int i = 0; i < row; i++)
        {
            Debug.Log("vec" + name + " i = " + i + " val = " + vec[i]);
        }
    }
    void DebugMatrix(int row, int col, float[] mat, string name)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                Debug.Log("Matrix" + name + " i = " + i + " j = " + j + " val = " + mat[i * col + j]);
            }
        }
    }
    void GravityStep()
    {
        for (int i = 0; i < node_num; i++)
        {
            float dis = Vector3.Distance(node_pos[i], spherePos);
            if (dis <= sphereRadius + 0.01f)
            {
                f[i * 3 + 0] = 0;
                f[i * 3 + 1] = 0;
                f[i * 3 + 2] = 0;
            }
            else
            {
                f[i * 3 + 0] = 0;
                f[i * 3 + 1] = -0.01f;
                f[i * 3 + 2] = 0;
            }
        }
    }
    void CollisionStep()
    {
        for (int i = 0; i < node_num; i++)
        {
            float dis = Vector3.Distance(new Vector3(pos[i*3+0],pos[i*3+1],pos[i*3+2]), spherePos);
            if (dis <= sphereRadius + 0.01f)
            {
                pos[i * 3 + 0] = (pos[i * 3 + 0] - spherePos.x) / dis * sphereRadius + spherePos.x;
                pos[i * 3 + 1] = (pos[i * 3 + 1] - spherePos.y) / dis * sphereRadius + spherePos.y;
                pos[i * 3 + 2] = (pos[i * 3 + 2] - spherePos.z) / dis * sphereRadius + spherePos.z;
            }
        }
    }
    void computefdf()
    {
        float energy = 0;
        for(int ic = 0; ic < constraint_num;ic++)
        {
            int st = d_st[ic];
            int ed = d_ed[ic];
            float[] pos_diff = { pos[st * 3 + 0] - pos[ed * 3 + 0], pos[st * 3 + 1] - pos[ed * 3 + 1], pos[st * 3 + 2] - pos[ed * 3 + 2] };
            
            float absx = Mathf.Sqrt(pos_diff[0] * pos_diff[0] + pos_diff[1] * pos_diff[1] + pos_diff[2] * pos_diff[2]);
            energy += 0.5f * stiffness[ic] * (absx - rest[ic]) * (absx - rest[ic]);

            float[] hatx = { pos_diff[0] / absx, pos_diff[1] / absx, pos_diff[2] / absx };
            float[] jacobian =  { hatx[0] * stiffness[ic] * (absx - rest[ic]), hatx[1] * stiffness[ic] * (absx - rest[ic]), hatx[2] * stiffness[ic] * (absx - rest[ic]) };
            f[st * 3 + 0] -= jacobian[0];
            f[st * 3 + 1] -= jacobian[1];
            f[st * 3 + 2] -= jacobian[2];
            f[ed * 3 + 0] += jacobian[0];
            f[ed * 3 + 1] += jacobian[1];
            f[ed * 3 + 2] += jacobian[2];

            float[] xxt = {hatx[0] * hatx[0], hatx[0] * hatx[1],hatx[0] * hatx[2],
                                hatx[1] * hatx[0], hatx[1] * hatx[1],hatx[1] * hatx[2],
                                hatx[2] * hatx[0], hatx[2] * hatx[1],hatx[2] * hatx[2]};
            for(int j = 0;j < 3;j++)
            {
                for(int i = 0; i < 3;i++)
                {
                    int idx = j * 3 + i;
                    float hessian = stiffness[ic] * (iden3[idx] - rest[ic] / absx * (iden3[idx] - xxt[idx]));
                    df[(st * 3 + j) * row + st * 3 + i] -= hessian;
                    df[(st * 3 + j) * row + ed * 3 + i] += hessian;
                    df[(ed * 3 + j) * row + st * 3 + i] += hessian;
                    df[(ed * 3 + j) * row + ed * 3 + i] -= hessian;
                }
            }
        }
    }
    void Assemble()
    {

        dfdx_x = MatrixMultipleVec(row, row, df, pos);
        for (int j = 0; j < row;j++)
        {
            for(int i = 0; i < row;i++)
            {
                int idx = j * row + i;
                if(i == j)
                {
                    A[idx] = node_mass[i / 3] - dt * dt * df[idx];
                }else
                {
                    A[idx] =  - dt * dt * df[idx];
                }
                df[idx] = 0;
            }
            inertia[j] = node_mass[j/3] * (2 * pos[j] - pos_pre[j]);
        }
        for(int i = 0;i < row;i++)
        {
            rhs[i] = inertia[i] + dt * dt * f[i] - dt * dt * dfdx_x[i];
            f[i] = 0;
            pos_pre[i] = pos[i];
        }
        pos = solver.DenseSolver(row, A, pos, rhs);
    }
    void UpdateMesh()
    {
        for(int i = 0; i < node_num;i++)
        {
            node_pos[i] = new Vector3(pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]);
        }
        mesh.vertices = node_pos;
        mesh.RecalculateNormals();
    }
    private void Update()
    {
        GravityStep();
        computefdf();
        Assemble();
        CollisionStep();
        UpdateMesh();
    }
}
