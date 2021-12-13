using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ReductionManager : MonoBehaviour
{
    // Start is called before the first frame update
    const int data_length = 128;
    float[] data_cpu = new float[data_length];
    ComputeBuffer data_gpu;
    ComputeBuffer result_gpu;
    public ComputeShader reductionShader;
    int kernelIndex;

    uint sizeX, sizeY, sizeZ;
    void Start()
    {
        for (int i = 0; i < data_length; i++) data_cpu[i] = i;
        kernelIndex = reductionShader.FindKernel("Reduction1");
        data_gpu = new ComputeBuffer(data_length, sizeof(float));
        result_gpu = new ComputeBuffer(data_length, sizeof(float));
        data_gpu.SetData(data_cpu);
        reductionShader.SetBuffer(kernelIndex, "Source",data_gpu);
        reductionShader.SetBuffer(kernelIndex, "Result", result_gpu);
        reductionShader.GetKernelThreadGroupSizes(kernelIndex,out sizeX,
            out sizeY,out sizeZ);
        reductionShader.Dispatch(kernelIndex, (int)(data_length / sizeX), 1, 1);

        reductionShader.SetBuffer(kernelIndex, "Source", result_gpu);
        reductionShader.SetBuffer(kernelIndex, "Result", data_gpu);
        reductionShader.Dispatch(kernelIndex, 1, 1, 1);

        float[] result = new float[data_length];
        data_gpu.GetData(result);
        foreach(var eachResult in result)
        {
            Debug.Log(eachResult);
        }

    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
