using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Matrix2x2
{
    public float v00, v01, v10, v11;

    public Matrix2x2()
    {
        v00 = v01 = v10 = v11;
    }

    public Matrix2x2(float a,float b,float c,float d)
    {
        v00 = a;
        v01 = b;
        v10 = c;
        v11 = d;
    }
    public static Matrix2x2 operator + (Matrix2x2 a, Matrix2x2 b)
    {
        return new Matrix2x2(a.v00 + b.v00, a.v01 + b.v01,
                                a.v10 + b.v10, a.v11 + b.v11);
    }

    public static Matrix2x2 operator - (Matrix2x2 a, Matrix2x2 b)
    {
        return new Matrix2x2(a.v00 - b.v00, a.v01 - b.v01,
                                a.v10 - b.v10, a.v11 - b.v11);
    }

    public static Matrix2x2 operator *(Matrix2x2 a, float b)
    {
        return new Matrix2x2(a.v00 * b, a.v01 * b,
                                a.v10 * b, a.v11 * b);
    }

    public static Matrix2x2 operator *( float b, Matrix2x2 a)
    {
        return new Matrix2x2(a.v00 * b, a.v01 * b,
                                a.v10 * b, a.v11 * b);
    }

    public static Matrix2x2 mul(Matrix2x2 a,Matrix2x2 b)
    {
        Matrix2x2 res = new Matrix2x2();
        res.v00 = a.v00 * b.v00 + a.v01 * b.v10;
        res.v01 = a.v00 * b.v01 + a.v01 * b.v11;
        res.v10 = a.v10 * b.v00 + a.v11 * b.v10;
        res.v11 = a.v10 * b.v01 + a.v11 * b.v11;
        return res;
    }
    public static float det(Matrix2x2 a)
    {
        return a.v00 * a.v11 - a.v10 * a.v01;
    }
    public static Matrix2x2 inv(Matrix2x2 a)
    {
        float de_inv = 1.0f / det(a);
        return new Matrix2x2(a.v11 * de_inv, -a.v01 * de_inv,
                                -a.v10 * de_inv, a.v00 * de_inv);
    }
    public Matrix2x2 transpose()
    {
        return new Matrix2x2(v00,v10,v01,v11);
    }

    public float doubleInner()
    {
        return v00 * v00 + v01 * v01 + v10 * v10 + v11 * v11;
    }

    public static Matrix2x2 identity()
    {
        return new Matrix2x2(1, 0, 0, 1);
    }

    public void debug(string name, int ie)
    {

        //Debug.Log("at element " +  ie + " and " + name + " = " + v00 + " " + v01 + " " + v10 + " " + v11);
    }
}
