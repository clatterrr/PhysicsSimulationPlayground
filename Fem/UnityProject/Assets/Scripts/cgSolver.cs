using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class cgSolver 
{
    // Start is called before the first frame update

    float[] DenseMultiple(int row,float[] A,float[] b)
    {
        float[] res = new float[row];
        for(int j = 0;j < row;j++)
        {
            res[j] = 0;
            for(int i = 0;i < row;i++)
            {
                res[j] += A[j * row + i] * b[i];
            }
        }
        return res;
    }

    float DenseDot(int row, float[] a, float[] b)
    {
        float res = 0; ;
        for (int j = 0; j < row; j++)
        {
            res += a[j] * b[j];
        }
        return res;
    }
    public float[] DenseSolver(int row,float[] A,float[] x,float[] b)
    {
        float[] d = new float[row];
        float[] res = new float[row];
        float[] Ax = DenseMultiple(row, A, x);
        for(int i = 0;i < row;i++)
        {
            res[i] = b[i] - Ax[i];
        }
        int it = 0, it_max = 200;
        float rho = 0, beta, rho_old, alpha;
        float[] Ad = new float[row];
        rho_old = 1;
        while(it < it_max)
        {
            it += 1;
            rho = DenseDot(row,res, res);
            if(rho < 1e-8)
            {
                break;
            }
            beta = 0;
            if (it > 0) beta = rho / rho_old;
            for (int i = 0; i < row; i++) d[i] = res[i] + beta * d[i];
            Ad = DenseMultiple(row,A, d);
            alpha = rho / DenseDot(row, d, Ad);
            for (int i = 0; i < row; i++)
            {
                x[i] = x[i] + alpha * d[i];
                res[i] = res[i] - alpha * Ad[i];
            }
            rho_old = rho;
        }
        //Debug.Log("solve finished at " + it + " and res = " + rho);
        return x;
    }
    float[] Sparse(int row, float[] A, float[] x, float b)
    {
        return x;
    }
}
