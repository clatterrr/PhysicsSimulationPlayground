奇异值分解，在python的numpy库，c++的eigen库，CUDA库也有实现。但是如果需要自己写一个怎么办呢？

最常见的就是分解变形梯度，把旋转部分分离掉，以计算出真正的变形量，对变形梯度的奇异值分解可见于物质点法，有限元以及连续介质力学。图像处理和深度学习也常用奇异值分解

本篇并仅打算讨论3x3矩阵的奇异值分解。

奇异值分解是指将矩阵A分解为三个矩阵
$$
\bold A = \bold U \Sigma \bold V^T
$$
其中U 和V是左奇异向量矩阵和右奇异向量矩阵，Sigma为对角矩阵。并行也不用考虑共享内存什么的，无脑写就行了。

# 开源库主流算法介绍

我还是习惯先把能找到的开源代码甩上来。常用的有四种。

## Eigen

著名的Eigen库的奇异值分解就实现了jacobian旋转方法，如果你觉得Eigen库太难读，可以看看我使用python实现了jacobian旋转，很方便调试。很多开源代码在需要奇异值分解的时候也会使用eigen的奇异值分解方法。

## 论文1

Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations

这篇论文中的算法非常经典。

首先是这篇论文自己就开源了代码，可见http://pages.cs.wisc.edu/~sifakis/project_pages/svd.html。

AMD公司的开源库GPUopen中的FEMFX也实现了这里的代码，即https://github.com/GPUOpen-Effects/FEMFX/blob/master/amd_femfx/src/Common/FEMFXSvd3x3.cpp

Descent Methods for Elastic Body Simulation on the GPU 也开源了，可见https://web.cse.ohio-state.edu/~wang.3602/publications.html

另一个是个人开源的地址为https://github.com/benjones/quatSVD/blob/master/quatSVD.hpp

## 论文2

Implicit-shifted Symmetric QR Singular Value Decomposition of 3 × 3 Matrices 

中介绍的算法也很受欢迎。

用于实现物质点法的的宾夕法尼亚大学开源库ziran中就有这篇论文自己的开源代码，地址在https://github.com/penn-graphics-research/ziran2019/blob/master/Lib/Ziran/Math/Linear/ImplicitQRSVD.h。

斯坦福大学的开源库physbam也使用了这种方法，地址在https://github.com/cuix-github/PhysBAM/blob/521b27e299f8add7ec42bb128f9cf9add4812eaf/PhysBAM_Tools/Matrices/MATRIX_3X3.cpp#L50

bullet3库也是如此https://github.com/romanpunia/tomahawk/blob/373f080ff8552dc770885f69a67f89edb6c98e09/src/supplies/bullet3/LinearMath/btImplicitQRSVD.h

最后是使用unity计算着色器实现的https://github.com/vanish87/UnitySVDComputeShader

## VegaFem

接下来介绍vegafem库所使用的方法，它是先计算矩阵A的特征值和特征向量，然后就能很快计算出奇异值。而特征值的计算用到了QL算法和householder分解。地址在https://github.com/starseeker/VegaFEM/blob/6252395422f96695e5403adaedd104040d7a7679/libraries/minivector/mat3d.cpp#L79。

vegafem库还处理了当sigma太小的情况。详细请看代码。这个算法也是04年论文可逆有限元 G. Irving, J. Teran, and R. Fedkiw. Invertible Finite Elements for Robust Simulation of Large Deformation 所使用的算法。

# 论文2简介

Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations 方法大致是先计算特征值，然后用特征值得到奇异值。

计算特征值，先使用雅可比迭代方法计算特征值，也就是对角化
$$
\bold A^T \bold A
$$
接下来算出的qV就是四元数形式的特征向量了。

又比如
$$
\bold Q_{unscaled} = \begin{bmatrix} c_h^2 - s_h^2 & -2s_hc_h  & 0 \\ 2s_hc_h & c_h^2 - s_h^2 & 0 \\ 0 & 0 & c_h^2 + s_h^2\end{bmatrix} \\
= (c_h^2 + s_h^2)\begin{bmatrix} \cos\phi & - \sin \phi & 0  \\ \sin\phi &  \cos \phi & 0 \\ 0 & 0 &  1\end{bmatrix} = (c_h^2 + s_h^2) \bold Q
$$
做三遍，第一遍
$$
\bold Q^T \bold S \bold Q = \begin{bmatrix} c & s & 0 \\ -s & c & 0 \\ 0 & 0 & 1\end{bmatrix}\bold S \begin{bmatrix} c & -s & 0 \\ s & c & 0 \\ 0 & 0 & 1\end{bmatrix} = \\
= \begin{bmatrix} ccs_{11}+css_{12} + css_{12}+sss_{12} & -css_{11} - sss_{12} + ccs_{12} +  css_{22} & cs_{13} + ss_{23} \\ -css_{11} + ccs_{12} - sss_{12} + scs_{22} & sss_{11} - css_{12} - css_{12} + ccs_{22} & -ss_{13} + cs_{23} \\ cs_{13} + ss_{23} & -ss_{13} + cs_{23} & s_{33}\end{bmatrix}
$$
写成代码如下

```
s11 = a*(a*t_s11 + b*t_s21) + b*(a*t_s21 + b*t_s22)
s21 = a*(-b*t_s11 + a*t_s21) + b*(-b*t_s21 + a*t_s22)
s22 = -b*(-b*t_s11 + a*t_s21) + a*(-b*t_s21 + a*t_s22)
s31 = a*t_s31 + b*t_s32
s33 = t_s33
```

然而上面的代码并不好调试，因此我重新用python写一版方便调试的代码，比较简单并且效率不高。首先仍然在计算奇异值之前计算特征值，特征值计算用QR迭代和householder分解。

householder矩阵将矩阵A变化为上Hessburger矩阵，第一次householder变换将第一列除了第一行的之外的元素都变为零，而不改变第一列的长度。第二次householder变换将第二列除了第一行和第二行之外的元素都变为零，而不改变第二列的长度。如下下去

![image-20211211154215233](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211154215233.png)

最后经过下式即可算出Q和R

![image-20211211151144181](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211151144181.png)

代码如下

```
def QRdecomposition(A):
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = householder(A[i:, i])
        Q = np.dot(Q, H)
        A = np.dot(H, A)
    return Q, A
```

但是如何将矩阵的一列生成指定个数的零，并且不改变那一列的长度呢？我们就可以使用householder变换了。这部分我参考了张贤达的《矩阵分析与应用》的第四章。

假如我们有两个向量，一个是x  = [3,2]，一个是坐标轴v = [1,0]。我们可以把x向量分解为平行于坐标轴和垂直于坐标轴的两个向量
$$
\bold x = \bold x^{||} + \bold x^{\perp} = \begin{bmatrix} 3\\2\end{bmatrix} = \begin{bmatrix} 3\\0\end{bmatrix} + \begin{bmatrix} 0\\2\end{bmatrix}
$$
![image-20211211105804227](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211105804227.png)

上面虽然可以一眼看出来，但是很多复杂情况下需要依靠计算投影矩阵和正交投影矩阵来计算
$$
\bold P_{\bold v} =  \frac{\bold v \bold v^{T}}{\bold v^T \bold v} =  \begin{bmatrix} 1 & 0 \\ 0 & 0\end{bmatrix} / 1 = \begin{bmatrix} 1 & 0 \\ 0 & 0\end{bmatrix} \qquad \bold P_{\perp \bold v} = \bold I - \bold P_{\bold v} = \begin{bmatrix} 0 & 0 \\ 0 & 1\end{bmatrix}
$$
这样我们可以这样计算分解了
$$
\bold x = \bold x^{||} + \bold x^{\perp} = \bold P_{\bold v}\bold x + \bold P_{\perp \bold v}\bold x
$$
但是等等，仔细观察那个正交投影矩阵，它把x投影倒垂直于v的地方，因此如果我们要算x关于垂直v向量的对称向量，可以使用反射法则，但也可以直接用之前计算出的投影矩阵直接计算

![image-20211211110901041](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211110901041.png)
$$
\bold H_{\bold v} \bold x =  \bold P_{\perp \bold v}\bold x - \bold P_{\bold v}\bold x = \begin{bmatrix} 0 \\ 2 \end{bmatrix} - \begin{bmatrix} 3 \\ 0 \end{bmatrix} =  \begin{bmatrix} -3 \\ 2 \end{bmatrix}
$$
这就是Householder变换，也被称为初等反射。其中HouseHolder矩阵根据上面的公式计算如下
$$
\bold H_{\bold v}  =  \bold P_{\perp \bold v} - \bold P_{\bold v} = \bold I - 2\frac{\bold v \bold v^{T}}{\bold v^T \bold v} = \begin{bmatrix} -1 & 0 \\ 0 & 1\end{bmatrix}
$$


继续代换进去，结果是一样的。householder矩阵既能使一向量的某些元素变为零，又能保持该向量的长度不变。例如下图

![image-20211211151916243](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211151916243.png)

要使向量x = [3,4]的第二个元素变为零，也就是转变为稀疏向量
$$
-||\bold x|| \bold e_1 = -\begin{bmatrix} 5 \\ 0 \end{bmatrix}
$$
我们首先构建向量
$$
\bold v = \bold x + ||\bold x||\bold e_1 = \begin{bmatrix} 8 \\ 4\end{bmatrix}
$$
再按照之前的步骤，将x分解为平行于向量v和垂直于v的向量，最后即可得到
$$
\bold P_{\bold v} \bold x = \begin{bmatrix} 4 \\ 2\end{bmatrix} \qquad \bold P_{\bold v}^{\perp} \bold x = \begin{bmatrix} -1 \\ 2\end{bmatrix} \qquad \bold H_{\bold v} \bold x = \begin{bmatrix} -5 \\ 0\end{bmatrix}
$$

```
def householder(a):
    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    inner = np.dot(v, v)
    outter = np.dot(v[:, None], v[None, :])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H
```

householder旋转的兄弟——givens旋转也会在接下来用到。计算特征值的方法用到了QR分解，进行多次QR分解，每次更新Q和R，并判断R是否渐趋稳定。

```
def computeEigenValue(A):
    A_new = A.copy()
    R_old = np.zeros((3,3))
    for i in range(100):
        Q,R = QRdecomposition(A_new)
        term0 = R[0,0] + R[1,1] + R[2,2]
        term1 = R_old[0,0] + R_old[1,1] + R_old[2,2]
        R_old = R.copy()
        if abs(term0 - term1)<1e-10:
            break
        A_new = np.dot(R,Q)
    return R_old
```

搞定特征值和特征向量后，接下来就很简单了，V就是特征向量，sigma就是特征值的sqrt。

```
sigma = np.zeros((3,3))
sigma[0,0] = np.sqrt(abs(eigenValue[0,0]))
sigma[1,1] = np.sqrt(abs(eigenValue[1,1]))
sigma[2,2] = np.sqrt(abs(eigenValue[2,2]))

svd_v = eigenVector

svd_u[:,0] = np.dot(A,svd_v[0,:]) / sigma[0,0]
svd_u[:,1] = np.dot(A,svd_v[1,:]) / sigma[1,1]
svd_u[:,2] = np.dot(A,svd_v[2,:]) / sigma[2,2]
```



Given旋转

同样给出一个向量x，
$$
\bold x = \begin{bmatrix} x_1 \\ x_2\end{bmatrix} = \begin{bmatrix} 4 \\ 2\end{bmatrix}
$$
现在我希望通过一些变换，在不改变向量长度的情况下，将这个向量的x1变成零，应该怎么办？

当然是通过旋转了，假设向量的x的幅角为phi，如果将其逆时针旋转 (pi - phi)角度后，那么x1就是零了。

如果是希望将向量的x2变成零，则是逆时针旋转 (pi / 2 - phi)角度了，如下图

![image-20211211112301854](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211112301854.png)

其中Givens旋转矩阵如下
$$
\begin{bmatrix} \bold y_1 \\ \bold y_2\end{bmatrix} = \begin{bmatrix} \cos \theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}\begin{bmatrix} \bold x_1 \\ \bold x_2\end{bmatrix} = \bold G^T \bold x
$$
这个东西看样子是个几何旋转问题，但就像之前那样我们可以仅仅使用旋转，而将某个向量x中任何位置的数字变成零。



```
float2 GetGivensConventionalCS(float a, float b)
{
	float d = a * a + b * b;
	float c = 1;
	float s = 0;
	if (abs(d) > 0)
	{
		float t = rsqrt(d);
		c = a * t;
		s = - b * t;
	}
	return float2(c, s);
}
```

例如为了将之前的x向量的x2位变量变为0，那么我们先计算d = 3^2 + 4^2 = 25，然后快速反平方根算出t = 0.2，那么cos = 0.8, sin = -0.6，也就是
$$
\begin{bmatrix} c & -s \\ s & c\end{bmatrix}\begin{bmatrix} x_1 \\ x_2\end{bmatrix} = \begin{bmatrix} 4/5 & 3/5 \\ -3/5 & 4/5\end{bmatrix}\begin{bmatrix} 4 \\ 3\end{bmatrix} = \begin{bmatrix} 5 \\ 0\end{bmatrix}
$$
如果我们想要x的向量的x1变为零，只需要将上面算法第10行加个负号就行了。

```
s = b * t;
```

同样，如果需要应用到三维向量的话，

```
float3x3 G3_23(float c, float s, bool use_conventional = true)
{
	float2 cs = use_conventional ? GetGivensConventionalCS(c, s) : GetGivensUnConventionalCS(c, s);
	c = cs.x;
	s = cs.y;
	return float3x3(1, 0, 0,
		0, c, s,
		0, -s, c);
}

void Bidiagonalize(inout float3x3 U, inout float3x3 A, inout float3x3 V)
{
	float3x3 G = G3_23(A[1][0], A[2][0]);
	A = mul(transpose(G), A);
	U = mul(U, G);
	//checked
	
	CodeZerochasing(U, A, V);
}
```

比如一个三维矩阵
$$
\bold A = \begin{bmatrix} 5 & 2 & 3\\ 4 & 5 & 6 \\ 3 & 8 & 9\end{bmatrix}
$$
我们希望A矩阵左下角的元素变成零，同时又希望UAVt的值不变，那么可以计算得到
$$
\bold G = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 4/5 & -3/5 \\ 0 & 3/5 & 4/5\end{bmatrix}
$$
那么新计算出来的U和A的值就是，注意矩阵的转置哦。
$$
\hat{\bold A} = \bold G^T \bold A \qquad \hat {\bold U} = \bold G \bold U
$$
![image-20211211125538030](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211125538030.png)

这个given旋转常见的应用就是极分解

![image-20211211122731718](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211122731718.png)

```
void GetPolarDecomposition2D(in float2x2 A, out float2x2 R, out float2x2 S)
{
	R = float2x2(0, 0, 0, 0);
	S = float2x2(0, 0, 0, 0);
	float x = A[0][0] + A[1][1];
	float y = A[1][0] - A[0][1];
	float d = sqrt(x*x + y*y);
	float c = 1;
	float s = 0;
	R = G2(c, s);
	if (abs(d) > EPSILON)
	{
		d = 1.0f / d;
		R = G2(x * d, -y * d);
	}
	S = mul(transpose(R), A);
}
```

搞定极分解后，就可实现2x2的奇异值分解了

![image-20211211122925424](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211122925424.png)

```
void GetSVD2D(in float2x2 A, out float2x2 U, out float2 D, out float2x2 V)
{
	U = float2x2(0, 0, 0, 0);
	D = float2(0, 0);
	V = float2x2(0, 0, 0, 0);

	float2x2 R = float2x2(0, 0, 0, 0);
	float2x2 S = float2x2(0, 0, 0, 0);

	GetPolarDecomposition2D(A, R, S);

	float c = 1;
	float s = 0;

	if (abs(S[0][1]) < EPSILON)
	{
		D[0] = S[0][0];
		D[1] = S[1][1];
	}
	else
	{
		float taw = 0.5f * (S[0][0] - S[1][1]);
		float w = sqrt(taw * taw + S[0][1] * S[0][1]);
		float t = taw > 0 ? S[0][1] / (taw + w) : S[0][1] / (taw - w);

		c = rsqrt(t*t + 1);
		s = -t * c;

		D[0] = c*c *S[0][0] - 2 * c*s*S[0][1] + s*s*S[1][1];
		D[1] = s*s *S[0][0] + 2 * c*s*S[0][1] + c*c*S[1][1];

	}

	if (D[0] < D[1])
	{
		float temp = D[0];
		D[0] = D[1];
		D[1] = temp;
		V = G2(-s, c);
	}
	else
	{
		V = G2(c, s);
	}

	U = mul(R, V);
}

```

上三角化

![image-20211211123734909](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20211211123734909.png)

```
void Bidiagonalize(inout float3x3 U, inout float3x3 A, inout float3x3 V)
{
	float3x3 G = G3_23(A[1][0], A[2][0]);
	A = mul(transpose(G), A);
	U = mul(U, G);
	//checked
	
	CodeZerochasing(U, A, V);
}
void CodeZerochasing(inout float3x3 U, inout float3x3 A, inout float3x3 V)
{
	float3x3 G = G3_12(A[0][0], A[1][0]);
	A = mul(transpose(G), A);
	U = mul(U, G);
	//checked
		
	float c = A[0][1];
	float s = A[0][2];
	if (abs(A[1][0]) > EPSILON)
	{
		c = A[0][0] * A[0][1] + A[1][0] * A[1][1];
		s = A[0][0] * A[0][2] + A[1][0] * A[1][2];
	}

	G = G3_23(c, s);
	A = mul(A, G);
	V = mul(V, G);
	//checked;
	
	G = G3_23(A[1][1], A[2][1]);
	A = mul(transpose(G), A);
	U = mul(U, G);
	//checked
}
```

