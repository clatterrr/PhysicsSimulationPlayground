using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fast : MonoBehaviour
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
    float[] node_mass = new float[node_num];
    Vector2[] uvs = new Vector2[node_num];

    const int stretch_num = (node_row_num - 1) * node_row_num * 2;
    const int shear_num = (node_row_num - 1) * (node_row_num - 1) * 2;
    const int bend_num = (node_row_num - 2) * node_row_num * 2;
    const int constraint_num = stretch_num + shear_num + bend_num;

    float[] stiffness = new float[constraint_num];

    const int row = node_num * 3;
    float[] M = new float[row * row];
    float[] L = new float[row * row];
    const int col = constraint_num * 3;
    float[] J = new float[row * col];
    float[] inertia = new float[row];
    float[] rhs = new float[row];
    float[] Jd = new float[row];
    float[] My = new float[row];
    float[] f_ext = new float[row];
    float[] pos = new float[row];
    float[] pos_pre = new float[row];

    int[] d_st = new int[constraint_num];
    int[] d_ed = new int[constraint_num];
    float[] d_val = new float[constraint_num * 3];
    float[] rest = new float[constraint_num];
    float dt = 1f;

    int cnt = 0;
    float[] Q = new float[row * row];
    float[] ch_L = new float[row * row];
    void Start()
    {
        meshFilter = GetComponent<MeshFilter>();
        mesh = new Mesh();
        meshFilter.mesh = mesh;
        drawTriangle();
        InitConstraint();
        CholeskyDecomp();
    
        spherePos = sphereCollision.transform.position;
        sphereRadius = sphereCollision.transform.localScale.x * 0.6f;
        for(int i = 0;i < row;i++)
        {
            pos_pre[i] = pos[i];
        }
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
                pos[idx * 3 + 0] = node_pos[idx].x;
                pos[idx * 3 + 1] = node_pos[idx].y;
                pos[idx * 3 + 2] = node_pos[idx].z;
                node_mass[idx] = 1;
                uvs[idx] = new Vector2((float)i / (node_row_num - 1), (float)j / (node_row_num - 1)) ;
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
    void ConstructMatrix(int st,int ed)
    {
        d_st[cnt] = st;
        d_ed[cnt] = ed;
        rest[cnt] = Vector3.Distance(node_pos[st], node_pos[ed]);
        stiffness[cnt] = 1;
        int idx00 = st * 3 * row + st * 3;
        int idx01 = st * 3 * row + ed * 3;
        int idx10 = ed * 3 * row + st * 3;
        int idx11 = ed * 3 * row + ed * 3;
        for(int  ir = 0; ir < 3;ir++)
        {
            L[idx00 + ir * row + ir] += stiffness[cnt];
            L[idx01 + ir * row + ir] -= stiffness[cnt];
            L[idx10 + ir * row + ir] -= stiffness[cnt];
            L[idx11 + ir * row + ir] += stiffness[cnt];
        }

        int idx0 = st * 3 * col + cnt * 3;
        int idx1 = ed * 3 * col + cnt * 3;
        for (int ir = 0; ir < 3; ir++)
        {
            J[idx0 + ir * col + ir] += stiffness[cnt];
            J[idx1 + ir * col + ir] -= stiffness[cnt];
        }
        cnt += 1;
    }
    void InitConstraint()
    {
        for(int i = 0; i < row * row;i ++)
        {
            L[i] = 0;
        }
        for (int i = 0; i < row * col; i++)
        {
            J[i] = 0;
        }

        for(int j = 0;j < node_row_num;j++)
        {
            for(int i = 0; i < node_row_num;i++)
            {
                int idx = j * node_row_num + i;
                // stretch
                if (i < node_row_num - 1) ConstructMatrix(idx, idx + 1);
                if (j < node_row_num - 1) ConstructMatrix(idx, idx + node_row_num);
                // bend
                if (i < node_row_num - 2) ConstructMatrix(idx, idx + 2);
                if (j < node_row_num - 2) ConstructMatrix(idx, idx + node_row_num * 2);
                // shear
                if((i < node_row_num - 1) && (j < node_row_num - 1))
                {
                    ConstructMatrix(idx, idx + 1 + node_row_num);
                    ConstructMatrix(idx + 1, idx + node_row_num);
                }
            }
        }
    }
    void CholeskyDecomp()
    {
        for (int i = 0; i < row * row; i++)
        {
            int row_now = i / row;
            int col_now = i % row;
            if(row_now == col_now)
            {
                Q[i] = node_mass[row_now/3] + dt * dt * L[i];
            }else
            {
                Q[i] = dt * dt * L[i];
            }

            ch_L[i] = 0;
        }
        float[] ch_v = new float[row];
        for(int j = 0;j < row;j++)
        {
            for(int i = j;i < row;i ++)
            {
                ch_v[i] = Q[i * row + j];
                for(int k = 0; k < j;k++)
                {
                    ch_v[i] -= ch_L[j * row + k] * ch_L[i * row + k];
                }
                ch_L[i * row + j] = ch_v[i] / Mathf.Sqrt(ch_v[j]);
            }
        }
    }
    float[] CholeskySolve(float[] rhs)
    {
        float[] resultTemp = new float[row];
        for(int i = 0;i < row;i++)
        {
            resultTemp[i] = rhs[i] / ch_L[i * row + i];
            for(int j = 0;j < i;j++)
            {
                resultTemp[i] -= ch_L[i * row + j] / ch_L[i * row + i] * resultTemp[j];
            }
        }
        float[] result = new float[row];
        for(int i = row - 1; i >= 0; i--)
        {
            result[i] = resultTemp[i] / ch_L[i * row + i];
            for(int j = i + 1;j < row;j++)
            {
                result[i] -= ch_L[j * row + i] / ch_L[i * row + i] * result[j];
            }
        }
        return result;
    }

    float[] MatrixMultipleVec(int size1, int size2,float[] Mat,float[] vec)
    {
        float[] res = new float[size1];
        for(int i = 0; i < size1;i ++)
        {
            res[i] = 0;
            for(int j = 0; j < size2;j++)
            {
                res[i] += Mat[i * size2 + j] * vec[j];
            }
        }

        return res;
    }
    void DebugVec(int row,  float[] vec, string name)
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
    void DebugMatrix(int row, int col,float[] mat,string name)
    {
        for(int i = 0; i <row;i++)
        {
            for(int j = 0; j < col;j++)
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
                f_ext[i*3+0] = 0;
                f_ext[i*3+1] = 0;
                f_ext[i*3+2] = 0;
            }
            else
            {
                f_ext[i * 3 + 0] = 0;
                f_ext[i * 3 + 1] = -0.01f;
                f_ext[i * 3 + 2] = 0;
            }
        }
    }
    void CollisionStep()
    {
        for (int i = 0; i < node_num; i++)
        {
            float dis = Vector3.Distance(new Vector3(pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]), spherePos);
            if (dis <= sphereRadius + 0.01f)
            {
                pos[i * 3 + 0] = (pos[i * 3 + 0] - spherePos.x) / dis * sphereRadius + spherePos.x;
                pos[i * 3 + 1] = (pos[i * 3 + 1] - spherePos.y) / dis * sphereRadius + spherePos.y;
                pos[i * 3 + 2] = (pos[i * 3 + 2] - spherePos.z) / dis * sphereRadius + spherePos.z;
            }
        }
    }
    void Step()
    {
        float energy = 0;
        for (int ic = 0;  ic < constraint_num; ic ++)
        {
            int st = d_st[ic];
            int ed = d_ed[ic];
            float diffx = pos[st * 3 + 0] - pos[ed * 3 + 0];
            float diffy = pos[st * 3 + 1] - pos[ed * 3 + 1];
            float diffz = pos[st * 3 + 2] - pos[ed * 3 + 2];
            float norm = Mathf.Sqrt(diffx * diffx + diffy * diffy + diffz * diffz);
            d_val[ic * 3 + 0] = rest[ic] * diffx / norm;
            d_val[ic * 3 + 1] = rest[ic] * diffy / norm;
            d_val[ic * 3 + 2] = rest[ic] * diffz / norm;
            energy += stiffness[ic] * (norm - rest[ic]) * (norm - rest[ic]) * 0.5f;
        }
        Jd = MatrixMultipleVec(row, col, J, d_val);
        for(int i = 0;i < row;i++)
        {
            rhs[i] = dt * dt * Jd[i] + node_mass[(int)(i/3)] * (2 * pos[i] - pos_pre[i]) + dt * dt * f_ext[i];
            pos_pre[i] = pos[i];
        }
        pos = CholeskySolve(rhs);

    }
    void UpdateMesh()
    {
        for (int i = 0; i < node_num; i++)
        {
            node_pos[i] = new Vector3(pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]);
        }
        mesh.vertices = node_pos;
        mesh.RecalculateNormals();
    }
    private void Update()
    {
        GravityStep();
        Step();
        CollisionStep();
        UpdateMesh();
    }
}
