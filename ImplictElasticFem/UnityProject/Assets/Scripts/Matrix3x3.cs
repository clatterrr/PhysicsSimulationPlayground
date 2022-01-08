using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*
public class Matrix3x3
{
    public float v00;
    public float v01;
    public float v02;
    public float v10;
    public float v11;
    public float v12;
    public float v20;
    public float v21;
    public float v22;

    public Matrix3x3()
    {
        this.v00 = 0;
        this.v01 = 0;
        this.v02 = 0;
        this.v10 = 0;
        this.v11 = 0;
        this.v12 = 0;
        this.v20 = 0;
        this.v21 = 0;
        this.v22 = 0;
    }

    public Matrix3x3(float v00, float v01, float v02,float v10, float v11,float v12,float v20,float v21,float v22)
    {
        this.v00 = v00;
        this.v01 = v01;
        this.v02 = v02;
        this.v10 = v10;
        this.v11 = v11;
        this.v12 = v12;
        this.v20 = v20;
        this.v21 = v21;
        this.v22 = v22;
    }

    public static Matrix3x3 identity()
    {
        Matrix3x3 res = new Matrix3x3(1.0f, 0.0f, 0.0f,0.0f, 1.0f,0.0f,0.0f,0.0f,1.0f);
        return res;
    }

    public static Matrix3x3 Multiple(Matrix3x3 A, Matrix3x3 B)
    {
        Matrix3x3 res = new Matrix3x3();
        res.v00 = A.v00 * B.v00 + A.v01 * B.v10 + A.v02 + B.v20;
        res.v01 = A.v00 * B.v01 + A.v01 * B.v11 + A.v02 + B.v21;
        res.v02 = A.v00 * B.v02 + A.v01 * B.v12 + A.v02 + B.v22;

        res.v10 = A.v10 * B.v00 + A.v11 * B.v10 + A.v12 + B.v20;
        res.v11 = A.v10 * B.v01 + A.v11 * B.v11 + A.v12 + B.v21;
        res.v12 = A.v10 * B.v02 + A.v11 * B.v12 + A.v12 + B.v22;

        res.v20 = A.v20 * B.v00 + A.v21 * B.v10 + A.v22 + B.v20;
        res.v21 = A.v20 * B.v01 + A.v21 * B.v11 + A.v22 + B.v21;
        res.v22 = A.v20 * B.v02 + A.v21 * B.v12 + A.v22 + B.v22;

        return res;
    }

    public static Matrix3x3 Transpose(Matrix3x3 A)
    {
        Matrix3x3 res = new Matrix3x3();
        res.v00 = A.v00;
        res.v01 = A.v10;
        res.v02 = A.v20;
        res.v10 = A.v01;
        res.v11 = A.v11;
        res.v12 = A.v21;
        res.v20 = A.v02;
        res.v21 = A.v12;
        res.v22 = A.v22;
        return res;
    }


    public static Matrix3x3 operator *(Matrix3x3 A, float scaler)
    {
        Matrix3x3 res = new Matrix3x3();
        res.v00 = A.v00 * scaler;
        res.v01 = A.v01 * scaler;
        res.v02 = A.v02 * scaler;
        res.v10 = A.v10 * scaler;
        res.v11 = A.v11 * scaler;
        res.v12 = A.v12 * scaler;
        res.v20 = A.v20 * scaler;
        res.v21 = A.v21 * scaler;
        res.v22 = A.v22 * scaler;
        return res;
    }

    public static Matrix3x3 operator +(Matrix3x3 A, Matrix3x3 B)
    {
        Matrix3x3 res = new Matrix3x3();
        res.v00 = A.v00 + B.v00;
        res.v01 = A.v01 + B.v01;
        res.v02 = A.v02 + B.v02;
        res.v10 = A.v10 + B.v10;
        res.v11 = A.v11 + B.v11;
        res.v12 = A.v12 + B.v12;
        res.v20 = A.v20 + B.v20;
        res.v21 = A.v21 + B.v21;
        res.v22 = A.v22 + B.v22;
        return res;
    }

    public static Matrix3x3 operator -(Matrix3x3 A, Matrix3x3 B)
    {
        Matrix3x3 res = new Matrix3x3();
        res.v00 = A.v00 - B.v00;
        res.v01 = A.v01 - B.v01;
        res.v02 = A.v02 - B.v02;
        res.v10 = A.v10 - B.v10;
        res.v11 = A.v11 - B.v11;
        res.v12 = A.v12 - B.v12;
        res.v20 = A.v20 - B.v20;
        res.v21 = A.v21 - B.v21;
        res.v22 = A.v22 - B.v22;
        return res;
    }
}
*/

public struct Matrix3x3
{
    public float v00;
    public float v01;
    public float v02;
    public float v10;
    public float v11;
    public float v12;
    public float v20;
    public float v21;
    public float v22;

    public Matrix3x3(float v00, float v01, float v02, float v10, float v11, float v12, float v20, float v21, float v22)
    {
        this.v00 = v00;
        this.v01 = v01;
        this.v02 = v02;
        this.v10 = v10;
        this.v11 = v11;
        this.v12 = v12;
        this.v20 = v20;
        this.v21 = v21;
        this.v22 = v22;
    }
    public static Matrix3x3 identity()
    {
        Matrix3x3 res = new Matrix3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
        return res;
    }

    public static Matrix3x3 Multiple(Matrix3x3 A, Matrix3x3 B)
    {
        Matrix3x3 res;
        res.v00 = A.v00 * B.v00 + A.v01 * B.v10 + A.v02 * B.v20;
        res.v01 = A.v00 * B.v01 + A.v01 * B.v11 + A.v02 * B.v21;
        res.v02 = A.v00 * B.v02 + A.v01 * B.v12 + A.v02 * B.v22;

        res.v10 = A.v10 * B.v00 + A.v11 * B.v10 + A.v12 * B.v20;
        res.v11 = A.v10 * B.v01 + A.v11 * B.v11 + A.v12 * B.v21;
        res.v12 = A.v10 * B.v02 + A.v11 * B.v12 + A.v12 * B.v22;

        res.v20 = A.v20 * B.v00 + A.v21 * B.v10 + A.v22 * B.v20;
        res.v21 = A.v20 * B.v01 + A.v21 * B.v11 + A.v22 * B.v21;
        res.v22 = A.v20 * B.v02 + A.v21 * B.v12 + A.v22 * B.v22;

        return res;
    }

    public Matrix3x3 Multiple(Matrix3x3 B)
    {
        Matrix3x3 res;
        res.v00 = this.v00 * B.v00 + this.v01 * B.v10 + this.v02 * B.v20;
        res.v01 = this.v00 * B.v01 + this.v01 * B.v11 + this.v02 * B.v21;
        res.v02 = this.v00 * B.v02 + this.v01 * B.v12 + this.v02 * B.v22;

        res.v10 = this.v10 * B.v00 + this.v11 * B.v10 + this.v12 * B.v20;
        res.v11 = this.v10 * B.v01 + this.v11 * B.v11 + this.v12 * B.v21;
        res.v12 = this.v10 * B.v02 + this.v11 * B.v12 + this.v12 * B.v22;

        res.v20 = this.v20 * B.v00 + this.v21 * B.v10 + this.v22 * B.v20;
        res.v21 = this.v20 * B.v01 + this.v21 * B.v11 + this.v22 * B.v21;
        res.v22 = this.v20 * B.v02 + this.v21 * B.v12 + this.v22 * B.v22;

        return res;
    }

    public static Matrix3x3 Transpose(Matrix3x3 A)
    {
        Matrix3x3 res;
        res.v00 = A.v00;
        res.v01 = A.v10;
        res.v02 = A.v20;
        res.v10 = A.v01;
        res.v11 = A.v11;
        res.v12 = A.v21;
        res.v20 = A.v02;
        res.v21 = A.v12;
        res.v22 = A.v22;
        return res;
    }

    //// return Amat._11 * Amat._22 * Amat._33 + Amat._12 * Amat._23 * Amat._31 + Amat._13 * Amat._21 * Amat._32 - Amat._13 * Amat._22 * Amat._31 - Amat._12 * Amat._21 * Amat._33 - Amat._11 * Amat._23 * Amat._32;
    public static float Determiant(Matrix3x3 A)
    {
        float res = 0;
        res += A.v00 * (A.v11 * A.v22 - A.v12 * A.v21);
        res += A.v01 * (A.v12 * A.v20 - A.v10 * A.v22);
        res += A.v02 * (A.v10 * A.v21 - A.v11 * A.v20);
        return res;
    }

    public static float squredNorm(Matrix3x3 A)
    {
        float res = 0;
        res += A.v00 * A.v00 + A.v01 * A.v01 + A.v02 * A.v02;
        res += A.v10 * A.v10 + A.v11 * A.v11 + A.v12 * A.v12;
        res += A.v20 * A.v20 + A.v21 * A.v21 + A.v22 * A.v22;
        return res;
    }

    public static Matrix3x3 Inverse(Matrix3x3 A)
    {
        float detinv = 1.0f / Determiant(A);
        Matrix3x3 res;
        res.v00 =( A.v11 * A.v22 - A.v12 * A.v21 ) *detinv;
        res.v01 = (A.v02 * A.v21 - A.v01 * A.v22) * detinv;
        res.v02 = (A.v01 * A.v12 - A.v02 * A.v11) * detinv;
        res.v10 = (A.v12 * A.v20 - A.v10 * A.v22) * detinv;
        res.v11 = (A.v00 * A.v22 - A.v02 * A.v20) * detinv;
        res.v12 = (A.v02 * A.v10 - A.v00 * A.v12) * detinv;
        res.v20 = (A.v10 * A.v21 - A.v11 * A.v20) * detinv;
        res.v21 = (A.v01 * A.v20 - A.v00 * A.v21) * detinv;
        res.v22 = (A.v00 * A.v11 - A.v01 * A.v10) * detinv;
        return res;
    }

    public static Matrix3x3 operator *(Matrix3x3 A, float scaler)
    {
        Matrix3x3 res;
        res.v00 = A.v00 * scaler;
        res.v01 = A.v01 * scaler;
        res.v02 = A.v02 * scaler;
        res.v10 = A.v10 * scaler;
        res.v11 = A.v11 * scaler;
        res.v12 = A.v12 * scaler;
        res.v20 = A.v20 * scaler;
        res.v21 = A.v21 * scaler;
        res.v22 = A.v22 * scaler;
        return res;
    }

    public static Matrix3x3 operator *(float scaler,Matrix3x3 A)
    {
        Matrix3x3 res;
        res.v00 = A.v00 * scaler;
        res.v01 = A.v01 * scaler;
        res.v02 = A.v02 * scaler;
        res.v10 = A.v10 * scaler;
        res.v11 = A.v11 * scaler;
        res.v12 = A.v12 * scaler;
        res.v20 = A.v20 * scaler;
        res.v21 = A.v21 * scaler;
        res.v22 = A.v22 * scaler;
        return res;
    }

    public static Matrix3x3 operator +(Matrix3x3 A, Matrix3x3 B)
    {
        Matrix3x3 res;
        res.v00 = A.v00 + B.v00;
        res.v01 = A.v01 + B.v01;
        res.v02 = A.v02 + B.v02;
        res.v10 = A.v10 + B.v10;
        res.v11 = A.v11 + B.v11;
        res.v12 = A.v12 + B.v12;
        res.v20 = A.v20 + B.v20;
        res.v21 = A.v21 + B.v21;
        res.v22 = A.v22 + B.v22;
        return res;
    }

    public static Matrix3x3 operator -(Matrix3x3 A, Matrix3x3 B)
    {
        Matrix3x3 res;
        res.v00 = A.v00 - B.v00;
        res.v01 = A.v01 - B.v01;
        res.v02 = A.v02 - B.v02;
        res.v10 = A.v10 - B.v10;
        res.v11 = A.v11 - B.v11;
        res.v12 = A.v12 - B.v12;
        res.v20 = A.v20 - B.v20;
        res.v21 = A.v21 - B.v21;
        res.v22 = A.v22 - B.v22;
        return res;
    }

    public static void debug(Matrix3x3 A,string name,int idx)
    {
        Debug.Log("===== " + A.v00 + " = " + A.v01 + " = " + A.v02);
        Debug.Log(name + " ="+idx+"= " + A.v10 + " = " + A.v11 + " = " + A.v12);
        Debug.Log("===== " + A.v20 + " = " + A.v21 + " = " + A.v22);
    }

}