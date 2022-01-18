



假如你有一个弱小可怜又无助的单位向量V，
$$
\bold V = \begin{bmatrix} \sqrt{2} /2 \\ \sqrt{2}/2\end{bmatrix}
$$
这个向量V的长度当然是1。而现在它被迫被矩阵A 乘了，旋转了一些，也伸缩了一些，现在它可能不是单位向量了。

但是我们对这个向量旋转的部分不感兴趣。因为这个向量可能是一段弹簧，三角形的一个边，或两个粒子之间的距离。而旋转是刚体运动。这个向量无论怎么旋转，都不会产生能量，也不会产生力。只有当这个向量伸缩的时候，才会产生能量和力，所以我们想知道这个向量究竟伸缩了多少。

我们可以使用极分解(polar decomposition)，也就是
$$
\bold A = \bold R \bold S
$$
但是它有缺点。





上面的特征值分解，对矩阵有着较高的要求，它需要被分解的矩阵AA为实对称矩阵，但是现实中，我们所遇到的问题一般不是实对称矩阵。

因此我们可以用奇异值分解，
$$
\bold A\bold V = \Sigma\bold U
$$

在目前的各种开源库中，特别是物理或几何领域，三维奇异值分解主要有四种方法。但是先介绍二维奇异值分解

# 二维奇异值分解

```
import numpy as np

def G2(c,s):
    return np.array([[c,s],[-s,c]])

def PolarDecompostion(A):
    R = np.zeros((2,2))
    S = np.zeros((2,2))
    x = A[0,0] + A[1,1]
    y = A[1,0] - A[0,1]
    d = np.sqrt(x*x + y*y)
    R = G2(1,0)
    if abs(d) > 1e-10:
        d = 1 / d
        R = G2(x*d,-y*d)
    S = np.dot(R.T,A)
    return R,S



A = np.array([[3.0,2.0],[1.0,4.0]])
u0,s0,vt0 = SVD2x2(A)
u1,s1,vt1 = np.linalg.svd(A)
```



# 方法一：先求特征值再求奇异值



## 使用方法一的缺点

当矩阵A 是奇异的时候，比如
$$
\bold A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0\end{bmatrix}
$$
那么就会为零的特征值，此时无法正确计算出左奇异矩阵U，因为左奇异矩阵的计算代码如下

```
    svd_u[:,0] = np.dot(A,svd_v[0,:]) / sigma[0,0]
    svd_u[:,1] = np.dot(A,svd_v[1,:]) / sigma[1,1]
    svd_u[:,2] = np.dot(A,svd_v[2,:]) / sigma[2,2]
```



## 

# 方法二：Eigen



极分解就是将矩阵分为旋转矩阵和剪切矩阵。

Eigen 数学库所使用的方法

著名的Eigen库的奇异值分解就实现了jacobian旋转方法，如果你觉得Eigen库太难读，可以看看我使用python实现了jacobian旋转，很方便调试。很多开源代码在需要奇异值分解的时候也会使用eigen的奇异值分解方法。这种方法适用于任何维度的矩阵，不过原理较为复杂。

https://mathworld.wolfram.com/JacobiRotationMatrix.html

![image-20220115112210132](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220115112210132.png)

==================Adaptive Jacobi method for parallel singular value decompositions  

![image-20220114203551422](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220114203551422.png)



```
rotate_s = 0
    rotate_c = 0
    if abs(d) < considerAsZero:
        rotate_s = 0
        rotate_c = 1
    else:
        u = t / d
        tmp = np.sqrt(1 + u*u)
        rotate_s = 1 / tmp
        rotate_c = u / tmp
```

![image-20220114205607464](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220114205607464.png)



```
def makeJacobi(x,y,z):
    deno = 2 * abs(y)
    m_c = 1
    m_s = 0
    if deno > considerAsZero:
        tau = (x - z) / deno
        w = np.sqrt(tau * tau + 1)
        t = 0
        if tau > 0:
            t = 1 / (tau + w)
        else:
            t = 1 / (tau - w)
        n = 1 / np.sqrt(t*t + 1)
        m_s = - np.sign(t) * y / abs(y) * abs(t) * n
        m_c = n
    return np.array([m_c,m_s])
```

![image-20220114203918923](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220114203918923.png)

====================D:\图形学书籍\svd\BLOCK-JACOB1SVD ALGORITHMS FOR.pdf

Dynamic ordering for a parallel block-Jacobi SVD algorithm  

![image-20220115095952916](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220115095952916.png)

=================D:\图形学书籍\svd\Two-sided hyperbolic singular value.pdf

![image-20220115174936947](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220115174936947.png)

# 方法三

Computing the Singular Value Decomposition of 3 x 3 matrices with minimal branching and elementary floating point operations

## QR given 旋转

经过householder 变换后，一个向量可以只剩下一个元素不为零，且这个元素就是向量的长度，而其它的元素都是零。 



![image-20220114144822374](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220114144822374.png)

```
import numpy as np

EPSILON = 1e-10

def QRGivens(a,b):
    c = 1
    s = 0
    if b == 0:
        return c,s
    else:
        if abs(b) > abs(a):
            tau = - a / b
            s = 1 / np.sqrt(1 + tau * tau)
            c = s * tau
        else:
            tau = - b / a
            c = 1 / np.sqrt(1 + tau * tau)
            s = c * tau
        return c,s
            

x = np.array([3.0,4.0])
c,s = QRGivens(x[0], x[1])
rot = np.array([[c,-s],[s,c]])
y = np.dot(rot,x)
```

参考

![image-20220114150930134](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220114150930134.png)



```
_gamma = 5.828427124 # FOUR_GAMMAt_sQUARED = sqrt(8)+3;
_cstar = 0.923879532 # cos(pi/8)
_sstar = 0.3826834323 # sin(pi/8)
EPSILON = 1e-6

def ApproxGivens(a11,a12,a22):
    ch = 2 * (a11 - a22)
    sh = a12
    w = 1.0 / np.sqrt(ch * ch + sh * sh)
    if _gamma * sh * sh < ch * ch:
        return w * ch, w * sh
    else:
        return _cstar,_sstar
```







# 方法四


$$
\bold A = \begin{bmatrix} 3 & 4 & 2 \\ 4 & 5 & 2 \\ 2 & 5 & 3\end{bmatrix}
$$
首先Bi




# 特征值

https://www.youtube.com/watch?v=mBcLRGuAFUk

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
def jacobiConjugation():
	...
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

# ReOrder

重新按降序排序

E:\mycode\Hair\PieletClothSim\ClothSim\Utils\MathUtility.h

# 快速

IPC

```
    void fastComputeSingularValues3d(const Eigen::Matrix3d& A,
        Eigen::Vector3d& singular_values)
    {
        using T = double;
        // decompose normal equations
        Eigen::Vector3d lambda;
        fastEigenvalues(A.transpose() * A, lambda);

        // compute singular values
        if (lambda(2) < 0)
            lambda = (lambda.array() >= (T)0).select(lambda, (T)0);
        singular_values = lambda.array().sqrt();
        if (A.determinant() < 0)
            singular_values(2) = -singular_values(2);
    }
```

```
    void fastSVD3d(const Eigen::Matrix3d& A,
        Eigen::Matrix3d& U,
        Eigen::Vector3d& singular_values,
        Eigen::Matrix3d& V)
    // 182 mults, 112 adds, 6 divs, 11 sqrts, 1 atan2, 1 sincos
    {
        using T = double;
        // decompose normal equations
        Eigen::Vector3d lambda;
        fastSolveEigenproblem(A.transpose() * A, lambda, V);

        // compute singular values
        if (lambda(2) < 0)
            lambda = (lambda.array() >= (T)0).select(lambda, (T)0);
        singular_values = lambda.array().sqrt();
        if (A.determinant() < 0)
            singular_values(2) = -singular_values(2);

        // compute singular vectors
        U.col(0) = A * V.col(0);
        T norm = U.col(0).norm();
        if (norm != 0) {
            T one_over_norm = (T)1 / norm;
            U.col(0) = U.col(0) * one_over_norm;
        }
        else
            U.col(0) << 1, 0, 0;
        Eigen::Vector3d v1_orthogonal = U.col(0).unitOrthogonal();
        Eigen::Matrix<T, 3, 2> other_v;
        other_v.col(0) = v1_orthogonal;
        other_v.col(1) = U.col(0).cross(v1_orthogonal);
        Eigen::Vector2d w = other_v.transpose() * A * V.col(1);
        norm = w.norm();
        if (norm != 0) {
            T one_over_norm = (T)1 / norm;
            w = w * one_over_norm;
        }
        else
            w << 1, 0;
        U.col(1) = other_v * w;
        U.col(2) = U.col(0).cross(U.col(1));
    }
```

# 参考





对角元素相减

![image-20220114184207880](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220114184207880.png)

=

A proof of convergence for two parallel Jacobi SVD algorithms,  

https://github.com/DiToMaVe/svd-jacobian

https://mathworld.wolfram.com/JacobiRotationMatrix.html

https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
