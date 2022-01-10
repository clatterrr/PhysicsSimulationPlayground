using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Jobs;
using Unity.Collections;
using Unity.Burst;
using Unity.Jobs;
using System.Threading.Tasks;
using System.IO;
using System;

public class Neohookean : MonoBehaviour
{

    // 顶点相关s
    int node_num;

    [NativeDisableParallelForRestriction]
    NativeArray<Vector3> node_pos;

    [NativeDisableParallelForRestriction]
    NativeArray<Vector3> node_force;

    [NativeDisableParallelForRestriction]
    NativeArray<int> node_upate;

    Vector3[] newpos;

    // 四面体相关
    int elem_num;
    [NativeDisableParallelForRestriction]
    NativeArray<int> elem_idx;

    [NativeDisableParallelForRestriction]
    NativeArray<float> elem_volume;

    [NativeDisableParallelForRestriction]
    NativeArray<Matrix3x3> elem_minv;

    [NativeDisableParallelForRestriction]
    NativeArray<Matrix3x3> elem_test;

    // 面相关，主要用于渲染
    int face_num;
    int[] face_idx;
    float[] face_uv;
    Vector3[] face_normal; 

    Mesh mesh;
    MeshFilter meshFilter;


    public string fileName = "creeperHigh.1";
    public float invmass = 1f;
    public float dt = 0.1f;
    public float mu = 50f;
    public float la = 50f;

    // https://www.raywenderlich.com/7880445-unity-job-system-and-burst-compiler-getting-started
    
    UpdateElementJob elementModificationJob;
    UpdateNodeJob nodeModificationJob;

    JobHandle elementModificationJobHandle;
    JobHandle nodeModificationJobHandle;
    void Start()
    {
        fileName = "cubeMed";
        invmass = 0.01f;
        dt = 0.01f;
        mu = 300f;
        la = 200f;
        mesh = new Mesh();
        ReadFile();
        mesh.vertices = newpos;
        mesh.triangles = face_idx;
        mesh.RecalculateBounds();
        meshFilter = GetComponent<MeshFilter>();
        meshFilter.mesh = mesh;
        //mesh.MarkDynamic();
    }

    void ReadFile()
    {
        string[] textTxt = File.ReadAllLines(Application.dataPath + "/TetModel/" + fileName + ".1.node");
        node_num = int.Parse(textTxt[0].Split(' ')[0]);
        node_pos = new NativeArray<Vector3>(node_num, Allocator.Persistent); 
        node_force = new NativeArray<Vector3>(node_num, Allocator.Persistent); 
        node_upate = new NativeArray<int>(node_num, Allocator.Persistent);
        newpos = new Vector3[node_num];
        Debug.Log(" node_num " + node_num);
        for (int i = 0;i < textTxt.Length;i++)
        {
            string[] splitTxt = textTxt[i+1].Split(' ', options: StringSplitOptions.RemoveEmptyEntries);
            if (splitTxt[0] == "#") break;
            node_pos[i] = new Vector3(float.Parse(splitTxt[1]), float.Parse(splitTxt[2]), float.Parse(splitTxt[3]));
            node_force[i] = Vector3.zero;
        }

        textTxt = File.ReadAllLines(Application.dataPath + "/TetModel/" + fileName + ".1.ele");
        elem_num = int.Parse(textTxt[0].Split(' ')[0]);
        elem_idx = new NativeArray<int>(elem_num * 4, Allocator.Persistent);
        elem_volume = new NativeArray<float>(elem_num, Allocator.Persistent);
        elem_minv = new NativeArray<Matrix3x3>(elem_num , Allocator.Persistent);
        elem_test = new NativeArray<Matrix3x3>(elem_num, Allocator.Persistent);
        for (int i = 0; i < textTxt.Length; i++)
        {
            string[] splitTxt = textTxt[i + 1].Split(' ', options: StringSplitOptions.RemoveEmptyEntries);
            if (splitTxt[0] == "#") break;
            elem_idx[i * 4 + 0] = int.Parse(splitTxt[1]);
            elem_idx[i * 4 + 1] = int.Parse(splitTxt[2]);
            elem_idx[i * 4 + 2] = int.Parse(splitTxt[3]);
            elem_idx[i * 4 + 3] = int.Parse(splitTxt[4]);

            Vector3 p0 = node_pos[elem_idx[i * 4 + 0]];
            Vector3 p1 = node_pos[elem_idx[i * 4 + 1]];
            Vector3 p2 = node_pos[elem_idx[i * 4 + 2]];
            Vector3 p3 = node_pos[elem_idx[i * 4 + 3]];

            float m11 = p1.x - p0.x, m12 = p2.x - p0.x, m13 = p3.x - p0.x;
            float m21 = p1.y - p0.y, m22 = p2.y - p0.y, m23 = p3.y - p0.y;
            float m31 = p1.z - p0.z, m32 = p2.z - p0.z, m33 = p3.z - p0.z;
            float det = m11 * m22 * m33 + m12 * m23 * m31 + m13 * m21 * m32 - m13 * m22 * m31 - m12 * m21 * m33 - m11 * m23 * m32;
            elem_volume[i] = det / 6;
            float dinv = 1.0f / det;
            elem_minv[i] = new Matrix3x3((m22 * m33 - m23 * m32) * dinv, (m13 * m32 - m12 * m33) * dinv, (m12 * m23 - m13 * m22) * dinv,
                                                          (m23 * m31 - m21 * m33) * dinv, (m11 * m33 - m13 * m31) * dinv, (m13 * m21 - m11 * m23) * dinv,
                                                          (m21 * m32 - m22 * m31) * dinv, (m12 * m31 - m11 * m32) * dinv, (m11 * m22 - m12 * m21) * dinv);
            //Matrix3x3.debug(elem_minv[i], "Dminv", i);
            elem_test[i] = new Matrix3x3();

        }

        textTxt = File.ReadAllLines(Application.dataPath + "/TetModel/" + fileName + ".1.face");
        face_num = int.Parse(textTxt[0].Split(' ')[0]);
        face_idx = new int[face_num * 3];
        face_normal = new Vector3[face_num];
        face_uv = new float[face_num * 3 * 2];
        for (int i = 0; i < textTxt.Length; i++)
        {
            string[] splitTxt = textTxt[i + 1].Split(' ', options: StringSplitOptions.RemoveEmptyEntries);
            if (splitTxt[0] == "#") break;
            face_idx[i * 3 + 0] = int.Parse(splitTxt[1]);
            face_idx[i * 3 + 2] = int.Parse(splitTxt[2]);
            face_idx[i * 3 + 1] = int.Parse(splitTxt[3]);

            Vector3 p0 = node_pos[face_idx[i * 3 + 0]];
            Vector3 p1 = node_pos[face_idx[i * 3 + 1]];
            Vector3 p2 = node_pos[face_idx[i * 3 + 2]];
            face_normal[i] = Vector3.Cross(p1 - p0, p2 - p0);
            face_uv[i * 6 + 0] = 0;
            face_uv[i * 6 + 1] = 0;
            face_uv[i * 6 + 2] = 1;
            face_uv[i * 6 + 3] = 0;
            face_uv[i * 6 + 4] = 0;
            face_uv[i * 6 + 5] = 1;
        }

        for(int i = 0;i < node_num;i++)
        {
           if(node_pos[i].y > 0.8f)
            {
                node_upate[i] = 1;
            }else if (node_pos[i].y < -0.8f)
            {
                node_upate[i] = 0;
            }else
            {
                node_upate[i] = 2;
            }
        }
    }

    [BurstCompile]
    private struct UpdateElementJob:IJobParallelFor
    {
        [ReadOnly] public NativeArray<Vector3> node_pos;
        public NativeArray<Vector3> node_force;

        [ReadOnly] public NativeArray<int> elem_idx;
        [ReadOnly]  public NativeArray<float> elem_volume;
        [ReadOnly]  public NativeArray<Matrix3x3> elem_Dminv;
        public NativeArray<Matrix3x3> elem_test;

        public int element_num;

        int e0, e1, e2, e3;

        Vector3 p0, p1, p2, p3, dx0, dx1, dx2;

        Matrix3x3 Ds, Dminv, F,Finv, FinvT, FtF,P, H;

        float J, logJ, energy, Ic, sumH;

        float la, mu, invMass;

        public void Execute(int ie)
        {
            if (ie >= element_num) return;

            e0 = elem_idx[ie * 4 + 0];
            e1 = elem_idx[ie * 4 + 1];
            e2 = elem_idx[ie * 4 + 2];
            e3 = elem_idx[ie * 4 + 3];

            p0 = node_pos[e0];
            p1 = node_pos[e1];
            p2 = node_pos[e2];
            p3 = node_pos[e3];

            dx0 = p1 - p0;
            dx1 = p2 - p0;
            dx2 = p3 - p0;

            Ds = new Matrix3x3(  dx0.x, dx1.x, dx2.x,
                                            dx0.y, dx1.y, dx2.y,
                                            dx0.z, dx1.z, dx2.z);

            Dminv = elem_Dminv[ie];
            elem_test[ie] = Dminv;
            F = Matrix3x3.Multiple(Ds, Dminv);

            Finv = Matrix3x3.Inverse(F);
            FinvT = Matrix3x3.Transpose(Finv);
            FtF = Matrix3x3.Multiple(Matrix3x3.Transpose(F), F);
            J = Matrix3x3.Determiant(F);
            logJ = Mathf.Log(J);
            Ic = FtF.v00 + FtF.v11 + FtF.v22;

           
            la = 2;
            mu = 2;
            invMass = 1f;

            energy = mu * 0.5f * (Ic - 3) - mu * logJ + la * 0.5f * logJ * logJ;

            P = mu * F - mu * FinvT + la * logJ * 0.5f * FinvT;

            

            H = -elem_volume[ie] * Matrix3x3.Multiple(P, Matrix3x3.Transpose(Dminv));

            

            sumH = invMass * Matrix3x3.squredNorm(H);
            sumH += invMass * (-H.v00 - H.v01 - H.v02) * (-H.v00 - H.v01 - H.v02);
            sumH += invMass * (-H.v10 - H.v11 - H.v12) * (-H.v10 - H.v11 - H.v12);
            sumH += invMass * (-H.v20 - H.v21 - H.v22) * (-H.v20 - H.v21 - H.v22);

            if (sumH < 1e-10) return;

            node_force[e1] += new Vector3(H.v00, H.v10, H.v20) * energy / sumH;
            node_force[e2] += new Vector3(H.v01, H.v11, H.v21) * energy / sumH;
            node_force[e3] += new Vector3(H.v02, H.v12, H.v22) * energy / sumH;
            node_force[e0] += new Vector3(-H.v00-H.v01-H.v02, -H.v10 - H.v11 - H.v12, -H.v20 - H.v21 - H.v22) * energy / sumH;

        }
    }

    [BurstCompile]
    private struct UpdateNodeJob : IJobParallelFor
    {
        public NativeArray<Vector3> node_pos;
        public NativeArray<Vector3> node_force;
        public NativeArray<int> node_update;

        public float dt;
        public void Execute(int ip)
        {
            if(node_update[ip] == 0)
            {
                
            }else if (node_update[ip] == 1)
            {
                node_pos[ip] += new Vector3(0, 0.01f, 0.0f);
            }
            else
            {
                node_pos[ip] += dt * node_force[ip];
            }
            node_force[ip] = Vector3.zero;
        }
    }
    private void OnDestroy()
    {
        node_pos.Dispose();
        node_force.Dispose();
        node_upate.Dispose();
        elem_idx.Dispose();
        elem_volume.Dispose();
        elem_minv.Dispose();
        elem_test.Dispose();
    }
    Matrix3x3 test;
    private void Update()
    {
        elementModificationJob = new UpdateElementJob()
        {
            node_force = node_force,
            node_pos =  node_pos,
            elem_Dminv = elem_minv,
            elem_idx = elem_idx,
            element_num = elem_num,
            elem_volume = elem_volume,
            elem_test = elem_test
        };
        elementModificationJobHandle = elementModificationJob.Schedule(node_num, 98);
        elementModificationJobHandle.Complete();

        for (int i = 0; i < elem_num; i++)
        {
            test = new Matrix3x3(elem_test[i].v00, elem_test[i].v01, elem_test[i].v02,
                                            elem_test[i].v10, elem_test[i].v11, elem_test[i].v12,
                                            elem_test[i].v20, elem_test[i].v21, elem_test[i].v22);
            //Matrix3x3.debug(test, "F", i);
        }
        for (int i = 0; i < node_num; i++)
        {
            //Debug.Log(" nodeforce = " + i + "  == " + node_force[i]);
        }

        nodeModificationJob = new UpdateNodeJob()
        {
            node_force = node_force,
            node_pos = node_pos,
            node_update = node_upate,
            dt = 0.1f
        };
        nodeModificationJobHandle = nodeModificationJob.Schedule(node_num, 98);

        nodeModificationJobHandle.Complete();

    }

    private void LateUpdate()
    {
        mesh.SetVertices(elementModificationJob.node_pos);

        mesh.RecalculateNormals();
    }
}
