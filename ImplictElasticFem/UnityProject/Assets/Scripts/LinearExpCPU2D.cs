using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LinearExpCPU2D : MonoBehaviour
{
    Mesh mesh;
    MeshFilter meshFilter;
    Vector3[] VerteicesArray = new Vector3[node_num];

    const int node_row_num = 10;
    const int node_num = node_row_num * node_row_num;
    const int element_row_num = node_row_num - 1;
    const int element_num = element_row_num * element_row_num * 2;

    Vector2[] node_pos = new Vector2[node_num];
    Vector2[] node_pos_correct = new Vector2[node_num];

    float[] element_Dm_inv = new float[element_num * 4];
    int[] element_idx = new int[element_num * 3];
    float[] element_area = new float[element_num];

    float lame_mu = 2;
    float lame_la = 2;
    float invMass = 1;
    float dt = 0.5f;
    float dx_space = 1f;//0.0f / node_row_num;
    void Start()
    {
        meshFilter = GetComponent<MeshFilter>();
        mesh = new Mesh();
        meshFilter.mesh = mesh;
        drawTriangle();
        InitFem();
    }
    void drawTriangle()
    {
        //We need two arrays one to hold the vertices and one to hold the triangles
        
        float space = 0.2f;
        for (int j = 0; j < node_row_num; j++)
        {
            for (int i = 0; i < node_row_num; i++)
            {
                int idx = j * node_row_num + i;
                VerteicesArray[idx] = new Vector3(i, j, 0) * dx_space;
                node_pos[idx] = new Vector2(i, j) * dx_space;
                node_pos_correct[idx] = new Vector2(0, 0);
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
    }

    void InitFem()
    {
        for(int ie = 0;ie < element_num;ie++)
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
            element_area[ie] =  det*0.5f;
            element_Dm_inv[ie * 4 + 0] = dX_11 * det_inv;
            element_Dm_inv[ie * 4 + 1] = -dX_01 * det_inv;
            element_Dm_inv[ie * 4 + 2] = -dX_10 * det_inv;
            element_Dm_inv[ie * 4 + 3] = dX_00 * det_inv;

        }
        node_pos[node_num - 1] += new Vector2(1, 1) * dx_space;
        //node_pos[1] = new Vector2(2, 0);
    }

    // Update is called once per frame
    void Update()
    {
        
        for (int ie = 0; ie < element_num; ie++)
        {
            Vector2 pos0 = node_pos[element_idx[ie * 3 + 0]];
            Vector2 pos1 = node_pos[element_idx[ie * 3 + 1]];
            Vector2 pos2 = node_pos[element_idx[ie * 3 + 2]];
            float dx_00 = pos1.x - pos0.x;
            float dx_01 = pos2.x - pos0.x;
            float dx_10 = pos1.y - pos0.y;
            float dx_11 = pos2.y - pos0.y;

            // Debug.Log(dx_00 + " " + dx_01 + " " + dx_10 + " " + dx_11);

            float dX_inv_00 = element_Dm_inv[ie * 4 + 0];
            float dX_inv_01 = element_Dm_inv[ie * 4 + 1];
            float dX_inv_10 = element_Dm_inv[ie * 4 + 2];
            float dX_inv_11 = element_Dm_inv[ie * 4 + 3];

            // deformation gradient
            // ÐÎ±äÌÝ¶È
            float F_00 = dx_00 * dX_inv_00 + dx_01 * dX_inv_10;
            float F_01 = dx_00 * dX_inv_01 + dx_01 * dX_inv_11;
            float F_10 = dx_10 * dX_inv_00 + dx_11 * dX_inv_10;
            float F_11 = dx_10 * dX_inv_01 + dx_11 * dX_inv_11;

            //Debug.Log("Deformation = "+ F_00 + " " + F_01 + " " + F_10 + " " +  F_11);

            float energy, piola_00,piola_01,piola_10,piola_11;

            int method = 0;
            
            if(method == 0)
            {

                float strain_00 = (F_00 * F_00 + F_10 * F_10 - 1) * 0.5f;
                float strain_01 = (F_00 * F_01 + F_10 * F_11) * 0.5f;
                float strain_10 = (F_01 * F_00 + F_11 * F_10) * 0.5f;
                float strain_11 = (F_01 * F_01 + F_11 * F_11 - 1) * 0.5f;

                //Debug.Log("strain = " + strain_00 + " " + strain_01 + " " + strain_10 + " " + strain_11);

                float strain_inner = strain_00 * strain_00 + strain_01 * strain_01
                                    + strain_10 * strain_10 + strain_11 * strain_11;

                float strain_trace = strain_00 + strain_11;

                energy = lame_mu * strain_inner + lame_la * 0.5f * strain_trace * strain_trace;

                //Debug.Log("Energy = " + energy);

                float pre_00 = 2.0f * lame_mu * strain_00 + lame_la * strain_trace;
                float pre_01 = 2.0f * lame_mu * strain_01;
                float pre_10 = 2.0f * lame_mu * strain_10;
                float pre_11 = 2.0f * lame_mu * strain_11 + lame_la * strain_trace;

                 piola_00 = F_00 * pre_00 + F_01 * pre_10;
                 piola_01 = F_00 * pre_01 + F_01 * pre_11;
                 piola_10 = F_10 * pre_00 + F_11 * pre_10;
                 piola_11 = F_10 * pre_01 + F_11 * pre_11;

                //Debug.Log("piola = " + piola_00 + " " + piola_01 + " " + piola_10 + " " + piola_11);
            }
            else
            {
                float strain_00 = F_00 - 1;
                float strain_01 = (F_01 + F_10) * 0.5f;
                float strain_10 = (F_10 + F_01) * 0.5f;
                float strain_11 = F_11 - 1;

                //Debug.Log("strain = " + strain_00 + " " + strain_01 + " " + strain_10 + " " + strain_11);

                float strain_inner = strain_00 * strain_00 + strain_01 * strain_01
                                    + strain_10 * strain_10 + strain_11 * strain_11;

                float strain_trace = strain_00 + strain_11;

                energy = lame_mu * strain_inner + lame_la * 0.5f * strain_trace * strain_trace;

                piola_00 = 2.0f * lame_mu * strain_00 + lame_la * (F_00 + F_11 - 2);
                piola_01 = 2.0f * lame_mu * strain_01;
                piola_10 = 2.0f * lame_mu * strain_10;
                piola_11 = 2.0f * lame_mu * strain_11 + lame_la * (F_00 + F_11 - 2);
            }

            

            //Debug.Log("piola = " + piola_00 + " " + piola_01 + " " + piola_10 + " " + piola_11);

            float area = element_area[ie];

            float dX_T_inv_00 = dX_inv_00;
            float dX_T_inv_10 = dX_inv_01;
            float dX_T_inv_01 = dX_inv_10;
            float dX_T_inv_11 = dX_inv_11;

            float H_00 = - area * (piola_00 * dX_T_inv_00 + piola_01 * dX_T_inv_10);
            float H_01 = - area * (piola_00 * dX_T_inv_01 + piola_01 * dX_T_inv_11);
            float H_10 = -area * (piola_10 * dX_T_inv_00 + piola_11 * dX_T_inv_10);
            float H_11 = -area * (piola_10 * dX_T_inv_01 + piola_11 * dX_T_inv_11);

            //Debug.Log("H = " + H_00 + " " + H_01 + " " + H_10 + " " + H_11);

            float gradC_00 = H_00,  gradC_01 = H_10;
            float gradC_10 = H_10, gradC_11 = H_11;
            float gradC_20 = -H_00 - H_10, gradC_21 = -H_10 - H_11;

            float sumGradC = (gradC_00 *gradC_00 + gradC_01 * gradC_01
                + gradC_10 * gradC_10 + gradC_11 * gradC_11 +
                gradC_20 * gradC_20 + gradC_21 * gradC_21);
            if (sumGradC < 1e-10) continue;

            float sumGradC_inv = 1.0f / sumGradC;

            node_pos_correct[element_idx[ie * 3 + 0]] += dt * energy * sumGradC_inv * invMass * new Vector2(gradC_20, gradC_21);
            node_pos_correct[element_idx[ie * 3 + 1]] += dt * energy * sumGradC_inv * invMass * new Vector2(gradC_00, gradC_01);
            node_pos_correct[element_idx[ie * 3 + 2]] += dt * energy * sumGradC_inv * invMass * new Vector2(gradC_10, gradC_11);
        }
        for (int j = 0; j < node_row_num; j++)
        {
            for (int i = 0; i < node_row_num; i++)
            {
                int idx = j * node_row_num + i;
                node_pos[idx] += node_pos_correct[idx];
                VerteicesArray[idx] = new Vector3(node_pos[idx].x, node_pos[idx].y, 0);
                node_pos_correct[idx] = new Vector2(0, 0);
            }
        }
        mesh.vertices = VerteicesArray;
        mesh.RecalculateNormals();
    }
}
