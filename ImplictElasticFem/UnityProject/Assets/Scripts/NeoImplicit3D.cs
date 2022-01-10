using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;

public class NeoImplicit3D : MonoBehaviour
{
    // Start is called before the first frame update
    int correctThread = 0;
    // �������s
    int node_num;
    float[] node_pos;
    float[] node_vel;
    float[] node_force;
    float[] node_force_new;
    int[] node_upate;
    private ComputeBuffer node_pos_buf;
    private ComputeBuffer node_vel_buf;
    private ComputeBuffer node_force_buf;
    private ComputeBuffer node_force_new_buf;
    private ComputeBuffer node_update_buf;

    // ���������
    int elem_num;
    int[] elem_idx;
    float[] elem_minv;
    float[] elem_volume;
    float[] elem_test;
    private ComputeBuffer elem_idx_buf;
    private ComputeBuffer elem_minv_buf;
    private ComputeBuffer elem_volume_buf;
    private ComputeBuffer elem_test_buf;

    // ����أ���Ҫ������Ⱦ
    int face_num;
    int[] face_idx;
    float[] face_uv;
    Vector3[] face_normal; // for rendering
    private ComputeBuffer face_idx_buf;
    private ComputeBuffer face_normal_buf;
    private ComputeBuffer face_uv_buf;

    // ������أ��������
    int krow;
    float[] Kmat;
    float[] Amat;
    float[] resvec;
    float[] xvec;
    float[] rhovec;
    float[] rho; // ֻ��һ��Ԫ��
    float[] dvec;
    float[] Advec;
    float[] AdAvec;
    float[] beta; // ֻ��һ��Ԫ��
    float[] alpha; // ֻ��һ��Ԫ��
    float[] rho1;// ֻ��һ��Ԫ��
    float[] AdA;
    float[] One;
    private ComputeBuffer Kmat_buf;
    private ComputeBuffer Amat_buf;
    private ComputeBuffer xvec_buf;
    private ComputeBuffer resvec_buf;
    private ComputeBuffer rhovec_buf;
    private ComputeBuffer rhovecsmall_buf;
    private ComputeBuffer dvec_buf;
    private ComputeBuffer Advec_buf;
    private ComputeBuffer AdAvec_buf;
    private ComputeBuffer beta_buf, alpha_buf, rho1_buf, AdA_buf, One_buf;
    bool CGdebug = false;

    public Material displayMaterial;
    public Texture texuture0;
    public Texture texuture1;
    public Texture texuture2;

    public ComputeShader stepCompute;
    public ComputeShader assemblyCompute;
    public ComputeShader addForceCompute;
    public ComputeShader correctCompute;

    public ComputeShader dotCompute;
    public ComputeShader reductionCompute;
    public ComputeShader addCompute;

    public ComputeShader BetaCompute;
    public ComputeShader AdCompute;
    public ComputeShader AlphaCompute;
    int reductionWARP_SIZE = 4;
    int MAT_WARP_SIZE = 8;
    int ELE_WARP_SIZE = 8;

    string fileName = "creeperHigh.1";
    float invmass = 10;
    float dt = 0.5f;
    float mu = 5f;
    float la = 5f;

    // �����ϰ��ĶԲߣ�0 Ϊʩ�ӷ���������������ˮ���ϸ���Ǳ��1 Ϊֱ�ӽ�������Ϊ�㣬���Ҳ��ٸ���λ�ã�������ײ����Ӳǽ�ڡ�
    int method = 0; 

    public enum NeoScheme
    {
        Custom,
        RigidCubeMedResolution,
        SoftCubeMedResolution,
        RigidCubeHighResolution,
        SoftCubeHighResolution,
        RigidChip,
        SoftChip,
        Bear,
        CreeperLowResolution,
        CreeperMedResolution,
        
    };
    public NeoScheme SelectAScene;
    
    void Start()
    {
        switch(SelectAScene)
        {
            case NeoScheme.Custom:break;
            case NeoScheme.RigidCubeMedResolution:
                {
                    fileName = "cubeMed";
                    invmass = 1;
                    dt = 0.1f;
                    mu = 20f;
                    la = 20f;
                    method = 1;
                    displayMaterial.SetTexture("Texture", texuture0);
                    break;
                }
            case NeoScheme.SoftCubeMedResolution:
                {
                    fileName = "cubeMed";
                    invmass = 5;
                    dt = 0.1f;
                    mu = 0.01f;
                    la = 0.02f;
                    method = 1;
                    displayMaterial.SetTexture("Texture", texuture0);
                    break;
                }
            case NeoScheme.RigidCubeHighResolution:
                {
                    fileName = "cubeHigh";
                    invmass = 10;
                    dt = 0.1f;
                    mu = 20f;
                    la = 20f;
                    method = 1;
                    displayMaterial.SetTexture("Texture", texuture2);
                    break;
                }
            case NeoScheme.SoftCubeHighResolution:
                {
                    fileName = "cubeHigh";
                    invmass = 10;
                    dt = 0.1f;
                    mu = 5f;
                    la = 5f;
                    method = 1;
                    displayMaterial.SetTexture("Texture", texuture2);
                    break;
                }
            case NeoScheme.RigidChip:
                {
                    fileName = "rod";
                    invmass = 1;
                    dt = 0.1f;
                    mu = 20f;
                    la = 20f;
                    method = 1;
                    displayMaterial.SetTexture("Texture", texuture0);
                    break;
                }
            case NeoScheme.SoftChip:
                {
                    fileName = "rod";
                    invmass = 1;
                    dt = 0.1f;
                    mu = 0.2f;
                    la = 0.2f;
                    method = 1;
                    displayMaterial.SetTexture("Texture", texuture0);
                    break;
                }
            case NeoScheme.CreeperLowResolution:
                {
                    fileName = "creeperLow";
                    invmass = 1;
                    dt = 0.1f;
                    mu = 20f;
                    la = 20f;
                    method = 1;
                    displayMaterial.SetTexture("Texture", texuture1);
                    break;
                }
            case NeoScheme.CreeperMedResolution:
                {
                    fileName = "creeperMed";
                    invmass = 1;
                    dt = 0.1f;
                    mu = 2f;
                    la = 2f;
                    displayMaterial.SetTexture("Texture", texuture1);
                    method = 1;
                    break;
                }
            case NeoScheme.Bear:
                {
                    fileName = "bear";
                    invmass = 1;
                    dt = 0.1f;
                    mu = 0.2f;
                    la = 0.2f;
                    method = 0;
                    displayMaterial.SetTexture("Texture", texuture2);
                    break;
                }
            default:break;
        }

        ReadFile();
        PrePare();



    }

    void ReadFile()
    {
        string[] textTxt = File.ReadAllLines(Application.dataPath + "/TetModel/" + fileName + ".1.node");
        node_num = int.Parse(textTxt[0].Split(' ')[0]);
        node_pos = new float[node_num * 3];
        node_vel = new float[node_num * 3];
        node_force = new float[node_num * 3];
        node_force_new = new float[node_num * 3];
        node_upate = new int[node_num * 3];
        for (int i = 0;i < textTxt.Length;i++)
        {
            string[] splitTxt = textTxt[i+1].Split(' ', options: StringSplitOptions.RemoveEmptyEntries);
            if (splitTxt[0] == "#") break;
            //for (int j = 0; j < splitTxt.Length; j++) Debug.Log(splitTxt[j]);
            node_pos[i * 3 + 0] = float.Parse(splitTxt[1]);
            node_pos[i * 3 + 1] = float.Parse(splitTxt[2]);
            node_pos[i * 3 + 2] = float.Parse(splitTxt[3]);
            node_vel[i * 3 + 0] = node_vel[i * 3 + 1] = node_vel[i * 3 + 2] = 0;
            node_force[i * 3 + 0] = node_force[i * 3 + 1] = node_force[i * 3 + 2] = 0;
            node_force_new[i * 3 + 0] = node_force_new[i * 3 + 1] = node_force_new[i * 3 + 2] = 0;
            node_upate[i * 3 + 0] = node_upate[i * 3 + 1] = node_upate[i * 3 + 2] = 1;
        }

        textTxt = File.ReadAllLines(Application.dataPath + "/TetModel/" + fileName + ".1.ele");
        elem_num = int.Parse(textTxt[0].Split(' ')[0]);
        elem_idx = new int[elem_num * 4];
        elem_volume = new float[elem_num];
        elem_minv = new float[elem_num * 9];
        elem_test = new float[elem_num * 9];
        for (int i = 0; i < textTxt.Length; i++)
        {
            string[] splitTxt = textTxt[i + 1].Split(' ', options: StringSplitOptions.RemoveEmptyEntries);
            if (splitTxt[0] == "#") break;
            elem_idx[i * 4 + 0] = int.Parse(splitTxt[1]);
            elem_idx[i * 4 + 1] = int.Parse(splitTxt[2]);
            elem_idx[i * 4 + 2] = int.Parse(splitTxt[3]);
            elem_idx[i * 4 + 3] = int.Parse(splitTxt[4]);

            Vector3 p0 = new Vector3( node_pos[elem_idx[i * 4 + 0] * 3 + 0], node_pos[elem_idx[i * 4 + 0] * 3 + 1], node_pos[elem_idx[i * 4 + 0] * 3 + 2]);
            Vector3 p1 = new Vector3(node_pos[elem_idx[i * 4 + 1] * 3 + 0], node_pos[elem_idx[i * 4 + 1] * 3 + 1], node_pos[elem_idx[i * 4 + 1] * 3 + 2]);
            Vector3 p2 = new Vector3(node_pos[elem_idx[i * 4 + 2] * 3 + 0], node_pos[elem_idx[i * 4 + 2] * 3 + 1], node_pos[elem_idx[i * 4 + 2] * 3 + 2]);
            Vector3 p3 = new Vector3(node_pos[elem_idx[i * 4 + 3] * 3 + 0], node_pos[elem_idx[i * 4 + 3] * 3 + 1], node_pos[elem_idx[i * 4 + 3] * 3 + 2]);

            float m11 = p1.x - p0.x, m12 = p2.x - p0.x, m13 = p3.x - p0.x;
            float m21 = p1.y - p0.y, m22 = p2.y - p0.y, m23 = p3.y - p0.y;
            float m31 = p1.z - p0.z, m32 = p2.z - p0.z, m33 = p3.z - p0.z;
            float det = m11 * m22 * m33 + m12 * m23 * m31 + m13 * m21 * m32 - m13 * m22 * m31 - m12 * m21 * m33 - m11 * m23 * m32;
            elem_volume[i] = det / 6;
            float dinv = 1.0f / det;
            elem_minv[i * 9 + 0] = (m22 * m33 - m23 * m32) * dinv;
            elem_minv[i * 9 + 1] = (m13 * m32 - m12 * m33) * dinv;
            elem_minv[i * 9 + 2] = (m12 * m23 - m13 * m22) * dinv;

            elem_minv[i * 9 + 3] = (m23 * m31 - m21 * m33) * dinv;
            elem_minv[i * 9 + 4] = (m11 * m33 - m13 * m31) * dinv;
            elem_minv[i * 9 + 5] = (m13 * m21 - m11 * m23) * dinv;

            elem_minv[i * 9 + 6] = (m21 * m32 - m22 * m31) * dinv;
            elem_minv[i * 9 + 7] = (m12 * m31 - m11 * m32) * dinv;
            elem_minv[i * 9 + 8] = (m11 * m22 - m12 * m21) * dinv;

            for (int j = 0; j < 9; j++)
            { elem_test[i * 9 + j] = 0; }
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

            Vector3 p0 = new Vector3(node_pos[face_idx[i * 3 + 0] * 3 + 0], node_pos[face_idx[i * 3 + 0] * 3 + 1], node_pos[face_idx[i * 3 + 0] * 3 + 2]);
            Vector3 p1 = new Vector3(node_pos[face_idx[i * 3 + 1] * 3 + 0], node_pos[face_idx[i * 3 + 1] * 3 + 1], node_pos[face_idx[i * 3 + 1] * 3 + 2]);
            Vector3 p2 = new Vector3(node_pos[face_idx[i * 3 + 2] * 3 + 0], node_pos[face_idx[i * 3 + 2] * 3 + 1], node_pos[face_idx[i * 3 + 2] * 3 + 2]);
            face_normal[i] = Vector3.Cross(p1 - p0, p2 - p0);
            face_uv[i * 6 + 0] = 0;
            face_uv[i * 6 + 1] = 0;
            face_uv[i * 6 + 2] = 1;
            face_uv[i * 6 + 3] = 0;
            face_uv[i * 6 + 4] = 0;
            face_uv[i * 6 + 5] = 1;
            //Debug.Log(face_idx[i * 3 + 0] + " " + face_idx[i * 3 + 1] + " " + face_idx[i * 3 + 2]);
        }

        //node_pos[0] += 5;
    }
    void PrePare()
    {
        node_pos_buf = new ComputeBuffer(node_num * 3, Marshal.SizeOf(typeof(float)));
        node_pos_buf.SetData(node_pos);
        face_idx_buf = new ComputeBuffer(face_num * 3, Marshal.SizeOf(typeof(int)));
        face_idx_buf.SetData(face_idx);
        face_normal_buf = new ComputeBuffer(face_num, Marshal.SizeOf(typeof(Vector3)));
        face_normal_buf.SetData(face_normal);
        face_uv_buf = new ComputeBuffer(face_num * 6, Marshal.SizeOf(typeof(float)));
        face_uv_buf.SetData(face_uv);
        displayMaterial.SetBuffer("_vertices", node_pos_buf);
        displayMaterial.SetBuffer("_idx", face_idx_buf);
        displayMaterial.SetBuffer("_normal", face_normal_buf);
        displayMaterial.SetBuffer("_uvs", face_uv_buf);

        node_vel_buf = new ComputeBuffer(node_num * 3, Marshal.SizeOf(typeof(float)));
        node_vel_buf.SetData(node_vel);
        node_force_buf = new ComputeBuffer(node_num * 3, Marshal.SizeOf(typeof(float)),ComputeBufferType.Raw);
        node_force_buf.SetData(node_force);
        node_force_new_buf = new ComputeBuffer(node_num * 3, Marshal.SizeOf(typeof(float)));
        node_force_new_buf.SetData(node_force_new);
        node_update_buf = new ComputeBuffer(node_num * 3, Marshal.SizeOf(typeof(int)));
        node_update_buf.SetData(node_upate);

        elem_idx_buf = new ComputeBuffer(elem_num * 4, Marshal.SizeOf(typeof(int)));
        elem_idx_buf.SetData(elem_idx);
        elem_minv_buf = new ComputeBuffer(elem_num * 9 , Marshal.SizeOf(typeof(float)));
        elem_minv_buf.SetData(elem_minv);
        elem_test_buf = new ComputeBuffer(elem_num * 9, Marshal.SizeOf(typeof(float)));
        elem_test_buf.SetData(elem_test);
        elem_volume_buf = new ComputeBuffer(elem_num, Marshal.SizeOf(typeof(float)));
        elem_volume_buf.SetData(elem_volume);

        krow = node_num * 3;
        Kmat = new float[krow * krow];
        Amat = new float[krow * krow];
        resvec = new float[krow];
        xvec = new float[krow];
        rhovec = new float[krow];
        dvec = new float[krow];
        Advec = new float[krow];
        AdAvec = new float[krow];
        beta = new float[1];
        alpha = new float[1];
        rho1 = new float[1];
        rho = new float[1];
        AdA = new float[1];
        One = new float[1];
        rho[0] =  alpha[0] = beta[0] = AdA[0] = 0;
        One[0] = rho1[0] = 1;
        for (int i = 0; i < krow * krow; i++) Kmat[i] = Amat[i] = 0;
        for (int i = 0; i < krow; i++) resvec[i] = xvec[i] = rhovec[i] = dvec[i] = Advec[i] = AdAvec[i] = 0;

        Kmat_buf = new ComputeBuffer(krow * krow, Marshal.SizeOf(typeof(float)),ComputeBufferType.Raw);
        Kmat_buf.SetData(Kmat);
        Amat_buf = new ComputeBuffer(krow * krow, Marshal.SizeOf(typeof(float)));
        Amat_buf.SetData(Amat);
        resvec_buf = new ComputeBuffer(krow , Marshal.SizeOf(typeof(float)));
        resvec_buf.SetData(resvec);
        xvec_buf = new ComputeBuffer(krow, Marshal.SizeOf(typeof(float)));
        xvec_buf.SetData(xvec);
        rhovec_buf = new ComputeBuffer(krow, Marshal.SizeOf(typeof(float)));
        rhovec_buf.SetData(rhovec);
        rhovecsmall_buf = new ComputeBuffer(1, Marshal.SizeOf(typeof(float)), ComputeBufferType.Raw);
        rhovecsmall_buf.SetData(rho);
        Advec_buf = new ComputeBuffer(krow, Marshal.SizeOf(typeof(float)));
        Advec_buf.SetData(Advec);
        dvec_buf = new ComputeBuffer(krow, Marshal.SizeOf(typeof(float)));
        dvec_buf.SetData(dvec);
        AdAvec_buf = new ComputeBuffer(krow, Marshal.SizeOf(typeof(float)));
        AdAvec_buf.SetData(AdAvec);

        beta_buf = new ComputeBuffer(1, Marshal.SizeOf(typeof(float)));
        beta_buf.SetData(beta);
        alpha_buf = new ComputeBuffer(1, Marshal.SizeOf(typeof(float)));
        alpha_buf.SetData(alpha);
        rho1_buf = new ComputeBuffer(1, Marshal.SizeOf(typeof(float)));
        rho1_buf.SetData(rho1);
        AdA_buf = new ComputeBuffer(1, Marshal.SizeOf(typeof(float)), ComputeBufferType.Raw);
        AdA_buf.SetData(AdA);
        One_buf = new ComputeBuffer(1, Marshal.SizeOf(typeof(float)));
        One_buf.SetData(One);
    }
    private void OnRenderObject()
    {
        displayMaterial.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Triangles, face_num * 3, 1);
    }
    void Step()
    {
        var fence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);
        int kernel = stepCompute.FindKernel("CSMain");
        stepCompute.SetBuffer(kernel, "ele_idx", elem_idx_buf);
        stepCompute.SetBuffer(kernel, "ele_minv", elem_minv_buf);
        stepCompute.SetBuffer(kernel, "ele_volume", elem_volume_buf);
        stepCompute.SetBuffer(kernel, "ele_test", elem_test_buf);
        stepCompute.SetBuffer(kernel, "node_pos", node_pos_buf);
        stepCompute.SetBuffer(kernel, "node_force", node_force_buf);
        stepCompute.SetBuffer(kernel, "Kmat", Kmat_buf);
        stepCompute.SetInt("krow", krow);
        stepCompute.SetFloat("mu", mu);
        stepCompute.SetFloat("la", la);
        stepCompute.Dispatch(kernel, elem_num / ELE_WARP_SIZE + correctThread, 1, 1);
        Graphics.WaitOnAsyncGraphicsFence(fence);
        
        /*
        elem_test_buf.GetData(elem_test);
        for(int i = 0;i < elem_num;i++)
        {
            //Debug.Log("idx = " + i);
            for(int j = 0;j < 9;j++)
            {
                //Debug.Log(elem_test[i * 9 + j]);
            }
        }
        node_force_buf.GetData(node_force);
        for (int i = 0; i < node_num; i++)
        {
            //Debug.Log("idx = " + i);
            for (int j = 0; j < 3; j++)
            {
                //Debug.Log(node_force[i * 3 + j]);
            }
        }
        Kmat_buf.GetData(Kmat);
        for(int j = 0;j < krow;j++)
        {
            for(int i = 0;i < krow;i++)
            {
                //Debug.Log("i = " + i + " j = " + j + " is " + Kmat[j * krow + i]);
            }
        }
        */
    }

    void CollisionAndGravity()
    {
        int kernel = addForceCompute.FindKernel("CSMain");
        addForceCompute.SetBuffer(kernel, "node_force", node_force_buf);
        addForceCompute.SetBuffer(kernel, "node_force_new", node_force_new_buf);
        addForceCompute.SetBuffer(kernel, "node_pos", node_pos_buf);
        addForceCompute.SetBuffer(kernel, "node_update", node_update_buf);
        addForceCompute.SetInt( "method", method);
        addForceCompute.Dispatch(kernel, krow / MAT_WARP_SIZE + 1, 1, 1);

        /*
        node_force_new_buf.GetData(node_force_new);
        for(int i = 0;i < krow;i++)
        {
            Debug.Log(i + " = " + node_force_new[i]);
        }
        */
    }

    void Assembly()
    {
        int kernel = assemblyCompute.FindKernel("CSMain");
        assemblyCompute.SetBuffer(kernel,"Kmat", Kmat_buf);
        assemblyCompute.SetBuffer(kernel, "node_force", node_force_new_buf);
        assemblyCompute.SetBuffer(kernel, "node_vel", node_vel_buf);
        assemblyCompute.SetBuffer(kernel, "Amat", Amat_buf);
        assemblyCompute.SetBuffer(kernel, "xvec", xvec_buf);
        assemblyCompute.SetBuffer(kernel, "resvec", resvec_buf);
        assemblyCompute.SetInt("krow", krow);
        assemblyCompute.SetFloat("invmass", invmass);
        assemblyCompute.SetFloat("dt", dt);
        assemblyCompute.Dispatch(kernel, krow / MAT_WARP_SIZE + 1, 1, 1);

        /*
        Amat_buf.GetData(Amat);
        for (int j = 0; j < krow; j++)
        {
            for (int i = 0; i < krow; i++)
            {
               //Debug.Log("i = " + i + " j = " + j + " is " + Amat[j * krow + i]);
            }
        }
        Kmat_buf.GetData(Kmat);
        for (int j = 0; j < krow; j++)
        {
            for (int i = 0; i < krow; i++)
            {
                //Debug.Log("i = " + i + " j = " + j + " is " + Kmat[j * krow + i]);
            }
        }
        resvec_buf.GetData(resvec);
        for(int i = 0; i < krow;i++)
        {
           // Debug.Log("resvec i = " + i + " = " + resvec[i]);
        }
        */
    }

    void CGsolve(bool first)
    {
        int kernel, WARP_SIZE;
        kernel = dotCompute.FindKernel("CSMain");
        dotCompute.SetBuffer(kernel,"xvec", resvec_buf);
        dotCompute.SetBuffer(kernel, "yvec", resvec_buf);
        dotCompute.SetBuffer(kernel, "result", rhovec_buf);
        dotCompute.Dispatch(kernel,krow / MAT_WARP_SIZE + 1, 1, 1);

        if(CGdebug)
        {
            rhovec_buf.GetData(rhovec);
            float realrho = 0;
            for (int i = 0; i < krow; i++)
            {
                Debug.Log("resvec i = " + i + " = " + rhovec[i]);
                realrho += rhovec[i];
            }
            Debug.Log("realrow" + realrho);
        }

        // rho = np.dot(resvec.T,resvec)
        WARP_SIZE = reductionWARP_SIZE;
        kernel = reductionCompute.FindKernel("ParallelReduction");
        reductionCompute.SetBuffer(kernel, "Source", rhovec_buf);
        reductionCompute.SetBuffer(kernel, "Result", rhovecsmall_buf);
        reductionCompute.Dispatch(kernel, krow / MAT_WARP_SIZE + 1, 1, 1);

        //rhovecsmall_buf.GetData(rho);

        kernel = BetaCompute.FindKernel("CSMain");
        BetaCompute.SetBuffer(kernel, "beta", beta_buf);
        BetaCompute.SetBuffer(kernel, "rho", rhovecsmall_buf);
        BetaCompute.SetBuffer(kernel, "rho1", rho1_buf);
        BetaCompute.SetBool("first", first);
        BetaCompute.Dispatch(kernel, 1, 1, 1);

        if(CGdebug)
        {
            beta_buf.GetData(beta);
            Debug.Log("beta = " + beta[0]);
        }

        
        // dvec = beta * dvec + resvec
        kernel = addCompute.FindKernel("CSMain");
        addCompute.SetBuffer(kernel, "xvec", dvec_buf);
        addCompute.SetBuffer(kernel, "yvec", resvec_buf);
        addCompute.SetBuffer(kernel,"xfactor",beta_buf);
        addCompute.SetBuffer(kernel, "yfactor", One_buf);
        addCompute.SetFloat("yfactor2", 1);
        addCompute.Dispatch(kernel, krow / MAT_WARP_SIZE + 1, 1, 1);

        if (CGdebug)
        {

            resvec_buf.GetData(dvec);
            for(int i = 0; i < krow;i++)
            {
                Debug.Log("dvec " + i + " = " + dvec[i]);
            }
            
        }
        
        // Advec = np.dot(Amat,dvec)
        kernel = AdCompute.FindKernel("CSMain");
        AdCompute.SetBuffer(kernel, "Amat", Amat_buf);
        AdCompute.SetBuffer(kernel, "dvec", dvec_buf);
        AdCompute.SetBuffer(kernel, "Advec", Advec_buf);
        AdCompute.SetInt("krow", krow);
        AdCompute.Dispatch(kernel, krow / MAT_WARP_SIZE + 1, 1, 1);

        if (CGdebug)
        {

            Advec_buf.GetData(Advec);
            for (int i = 0; i < krow; i++)
            {
                Debug.Log("Advec " + i + " = " + Advec[i]);
            }

        }

        // AdA = np.dot(dvec.T,Advec) 
        kernel = dotCompute.FindKernel("CSMain");
        dotCompute.SetBuffer(kernel, "xvec", Advec_buf);
        dotCompute.SetBuffer(kernel, "yvec", dvec_buf);
        dotCompute.SetBuffer(kernel, "result", AdAvec_buf);
        WARP_SIZE = 1;
        dotCompute.Dispatch(kernel, krow / WARP_SIZE, 1, 1);

        
        WARP_SIZE = reductionWARP_SIZE;
        kernel = reductionCompute.FindKernel("ParallelReduction");
        reductionCompute.SetBuffer(kernel, "Source", AdAvec_buf);
        reductionCompute.SetBuffer(kernel, "Result", AdA_buf);
        reductionCompute.Dispatch(kernel, krow / WARP_SIZE + 1, 1, 1);

        if (CGdebug)
        {
            AdA_buf.GetData(AdA);
            Debug.Log("AdA = " + AdA[0]);
        }

        kernel = AlphaCompute.FindKernel("CSMain");
        AlphaCompute.SetBuffer(kernel, "AdA", AdA_buf);
        AlphaCompute.SetBuffer(kernel, "rho1", rho1_buf);
        AlphaCompute.SetBuffer(kernel, "alpha", alpha_buf);
        AlphaCompute.Dispatch(kernel, 1, 1, 1);

        if (CGdebug)
        {
            alpha_buf.GetData(alpha);
            rho1_buf.GetData(rho1);
            Debug.Log("alpha = " + alpha[0] + " rho = " + rho1[0]);
        }

        // xvec = xvec + alpha * dvec
        kernel = addCompute.FindKernel("CSMain");
        addCompute.SetBuffer(kernel, "xvec", xvec_buf);
        addCompute.SetBuffer(kernel, "yvec", dvec_buf);
        addCompute.SetBuffer(kernel, "xfactor", One_buf);
        addCompute.SetBuffer(kernel, "yfactor", alpha_buf);
        addCompute.SetFloat("yfactor2", 1);
        addCompute.Dispatch(kernel, krow / MAT_WARP_SIZE + 1, 1, 1);

        if (CGdebug)
        {

            xvec_buf.GetData(xvec);
            for (int i = 0; i < krow; i++)
            {
                Debug.Log("xvec " + i + " = " + xvec[i]);
            }

        }

        // resvec = resvec - alpha * Advec
        kernel = addCompute.FindKernel("CSMain");
        addCompute.SetBuffer(kernel, "xvec", resvec_buf);
        addCompute.SetBuffer(kernel, "yvec", Advec_buf);
        addCompute.SetBuffer(kernel, "xfactor", One_buf);
        addCompute.SetBuffer(kernel, "yfactor", alpha_buf);
        addCompute.SetFloat("yfactor2", -1);
        addCompute.Dispatch(kernel, krow / MAT_WARP_SIZE + 1, 1, 1);

        if (CGdebug)
        {

            resvec_buf.GetData(resvec);
            for (int i = 0; i < krow; i++)
            {
                Debug.Log("resvec " + i + " = " + resvec[i]);
            }

        }
    }
    void CorrectPos()
    {
        int kernel = correctCompute.FindKernel("CSMain");
        correctCompute.SetFloat("dt", dt);
        correctCompute.SetBuffer(kernel, "xvec", xvec_buf);
        correctCompute.SetBuffer(kernel, "node_pos", node_pos_buf);
        correctCompute.SetBuffer(kernel, "node_vel", node_vel_buf);
        correctCompute.SetBuffer(kernel, "node_update", node_update_buf);
        correctCompute.Dispatch(kernel, krow / MAT_WARP_SIZE + 1, 1, 1);
    }
    void Update()
    {

        Step();

        CollisionAndGravity();

        Assembly();

        CGdebug = false;
        for (int cg = 0; cg < 100; cg++)
        {
            CGsolve(cg == 0);
        }

        CorrectPos();
    }
}
