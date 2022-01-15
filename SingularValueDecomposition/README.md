奇异值分解(singular value decompostion) 是一种非常常见的矩阵分解算法，主要用于将矩阵分解为伸缩矩阵和旋转矩阵。在物理或几何领域，最常用的就是对变形梯度的奇异值分解了。其它领域比如图像处理，深度学习，以及数据拟合也常用奇异值分解。不过本篇文章只讨论二维和三维的奇异值分解。



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

def SVD2x2(A):
    U = np.zeros((2,2))
    D = np.zeros((2))
    V = np.zeros((2,2))
    R,S = PolarDecompostion(A)
    c = 1
    s = 0
    if abs(S[0,1]) < 1e-10:
        D[0] = S[0,0]
        D[1] = S[1,1]
    else:
        taw = 0.5 * (S[0,0] - S[1,1])
        w = np.sqrt(taw * taw + S[0,1]*S[0,1])
        if taw > 0:
            t = S[0,1] / (taw + w)
        else:
            t = S[0,1] / (taw - w)
        c = 1.0 / np.sqrt(t * t + 1)
        s = - t * c
        D[0] = c*c*S[0,0] - 2*c*s*S[0,1] + s*s*S[1,1]
        D[1] = s*s*S[0,0] + 2*c*s*S[0,1] + c*c*S[1,1]
    if D[0] < D[1]:
        temp = D[0]
        D[0] = D[1]
        D[1] = temp
        V = G2(-s,c)
    else:
        V = G2(c,s)
    U = np.dot(R,V)
    return U,D,V

A = np.array([[3.0,2.0],[1.0,4.0]])
u0,s0,vt0 = SVD2x2(A)
u1,s1,vt1 = np.linalg.svd(A)
```



# 方法一：先求特征值再求奇异值

先回忆一下特征值分解是怎么回事
$$
\bold A \bold x = \lambda \bold x
$$
其中A 是任意矩阵，x是特征向量，lambda是特征值。稍微变化一下
$$
\bold A = \bold x \lambda \bold x^{-1}
$$
因此对于svd分解来说，我们可以做如下变化
$$
\bold A^T \bold A = (\bold U \Sigma \bold V^T)^T \bold U \bold \Sigma \bold V \\
= \bold V \Sigma \bold U^T \bold U \Sigma \bold V^T \\
= \bold V \Sigma^2 \bold V^T
$$
注意矩阵V 是旋转矩阵的一部分，是正交矩阵，它的转置等于逆。那么也就说，Vt 就是通过求特征向量即可得到，而sigma 则是特征值的sqrt。求特征值常用的方法是QR分解迭代，而QR分解常用的方法是householder变换。我们一步一步来。

## 使用householder变换

在攻打主线之前，先考虑一个简单的支线例子，假设我们有一个向量
$$
\bold x = \begin{bmatrix}3 \\ 2 \end{bmatrix}
$$
我们希望将它旋转到横着的第一根坐标轴上，也就是我们希望最后的结果是
$$
\bold H \bold x = -\begin{bmatrix}3.6 \\ 0 \end{bmatrix}
$$
那么如何算出旋转矩阵 H 呢？我们可以借助几何镜面反射的规律

![image-20220114121142965](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220114121142965.png)



如上图，黄色即是第一根根镜面反射轴，橙色虚线即是与镜面反射轴正交的第二根镜面反射轴。要算出旋转矩阵H ，我们只需要先把向量 x 投影到第一根黄色的镜面反射轴，然后再减去橙色虚线即可。

为此我们先计算出黄色的第一根镜面反射轴。为此我们先算出向量 x 的长度，并且e1轴上产生这样一个向量，也就是上图的橙色实线。然后再让橙色实线与原来向量x 相加，得到第一根镜面反射轴
$$
\bold v = \bold x + ||\bold x||\bold e_1 = \begin{bmatrix} 3 \\ 2 \end{bmatrix} + 3.6 \begin{bmatrix}1 \\ 0 \end{bmatrix} = \begin{bmatrix}6.6 \\ 2 \end{bmatrix}
$$
然后运用投影定理，将向量x 投影到这第一根黄色镜面反射轴上
$$
\bold P_v= \frac{\bold v \bold v^{T}}{\bold v^T \bold v} = 
\begin{bmatrix} 0.916 & 0.277 \\ 0.277 & 0.083\end{bmatrix}\qquad  \bold P_v \bold x = \frac{\bold v \bold v^{T}}{\bold v^T \bold v} \bold x = \begin{bmatrix}3.3 \\ 1 \end{bmatrix}
$$
这样一来Pvx 就和 向量 v 方向相同了，此时Pvx 也是下图蓝色虚线的一半。

![image-20220114123602581](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220114123602581.png)

接下来再算橙色虚线，也就是第二个镜面反射轴。由于第二根镜面反射轴与第一根正交，因此只需要用单位矩阵减去第一根投影变换矩阵就可得第二个投影变换矩阵。第二根镜面反射轴的向量也很好计算。
$$
\bold P_{\perp \bold v} = \bold I - \bold P_{\bold v} = \begin{bmatrix} 0.084 & -0.277 \\ -0.277 & 0.916\end{bmatrix} \qquad \bold P_{\perp \bold v} \bold x = \begin{bmatrix}-0.302 \\ 1 \end{bmatrix}
$$
接下就可得到Hx 的表达式
$$
\bold H \bold x = \bold x - 2 \bold P_v \bold x = \bold P_{\perp \bold v} \bold x - \bold P_v \bold x
$$
或者换一种写法
$$
\bold H_{\bold v}   = \bold I - 2\frac{\bold v \bold v^{T}}{\bold v^T \bold v}=  \bold P_{\perp \bold v} - \bold P_{\bold v}
$$
至此，householder 变换完成。写成代码如下

```
import numpy as np
def householder(a):
    v = a.copy()
    v[0] += np.linalg.norm(a)
    H = np.eye(a.shape[0])
    inner = np.dot(v, v)
    outter = np.dot(v[:, None], v[None, :])
    Pv = outter / inner
    Pvx = np.dot(Pv,x)
    Pv_ortho = np.eye(a.shape[0]) - Pv
    Pvx_or = np.dot(Pv_ortho,x)
    H -= (2 / inner) * outter
    return H
x = np.array([3,2],dtype = float)
h = householder(x)
hx = np.dot(h,x)
```

## 计算QR分解

QR分解将矩阵A 分解为正交矩阵和上三角矩阵
$$
\bold A = \bold Q \bold R
$$
将矩阵全写出来更清楚一些
$$
\bold A = \begin{bmatrix} * & * & * \\ * & * & * \\ * & * & *\end{bmatrix}\qquad  \bold R = \begin{bmatrix} * & * & * \\ 0 & * & * \\ 0 & 0 & *\end{bmatrix}
$$
但是矩阵A 既不是上三角矩阵，也不是正三角矩阵，怎么办呢？我们可以使用一个比较的粗鲁的几何办法，既然R是n行n列的上三角矩阵，那么也就是矩阵R第m 列，从(m+1)到n行都为零。而Q矩阵又是个正交矩阵，或者说是旋转矩阵，那么我们要对A 矩阵的第m列做的，不过就是把把第m 列视作一个向量，然后将它投影到第一根轴
$$
\bold A = \begin{bmatrix} 17 & 19 & 16 \\ 19 & 51 & 19 \\ 16 & 19 & 17\end{bmatrix} \qquad A_0 = \begin{bmatrix} 17 \\ 19
\\16 \end{bmatrix}
$$
例如对于上面的矩阵A，它的第一列向量是[17 19 16]^T，我们要想办法把算出一个旋转矩阵，把它投影到第一根轴上，并且保持长度不变。这就是householder 变换，上面已经介绍过了。把第一列投影到第一根轴上后，A 矩阵成了如下的样子
$$
\bold H _1\bold A = \begin{bmatrix} -30 & -53 & -30 \\ 0 & 21 & 0.41 \\ 0 & -5.46 & 1.35\end{bmatrix}
$$
接下来第二列怎么操作呢？很简单，我们直接忽略第一行和第一列的数字就行了，也就是我们只关注不是星号的数字
$$
\begin{bmatrix} * & * & * \\ * & 21 & 0.41 \\ * & -5.46 & 1.35\end{bmatrix} \qquad A_1 = \begin{bmatrix} 21 \\ -5.46\end{bmatrix}
$$
这样第一列又是A1了，接下来继续house holder 变换就完了。当进行到矩阵A的最后的第一列的时候，此时矩阵A 也变成了上三角矩阵R，而正交矩阵R 则由历次算出来的H 矩阵相乘得到
$$
\bold H_2 \bold H_1 \bold A = \bold R  \qquad \bold Q = \bold H_2 \bold H_1
$$
写成代码如下

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

## shift QR

上面的迭代QR分解有个缺点，就是当矩阵 A 是单位矩阵时，算出来的特征值是错的。



https://www.andreinc.net/2021/01/25/computing-eigenvalues-and-eigenvectors-using-qr-decomposition

```
def computeEigenValue(A):
    A_new = A.copy()
    R_old = np.zeros((3,3))
    for i in range(500):
        si = A_new[2,2] * np.eye(3)
        Q,R = QRdecomposition(A_new - si)
        term0 = R[0,0] + R[1,1] + R[2,2]
        term1 = R_old[0,0] + R_old[1,1] + R_old[2,2]
        R_old = R.copy()
        if abs(term0 - term1)<1e-10:
            break
        A_new = np.dot(R,Q) + si
    return A_new
```

=============D:\图形学书籍\图形学书籍\流体\Matrix Computations-Johns Hopkins University Press (2012).pdf

The theorem says that if we shift by an exact eigenvalue, then in exact arithmetic
deflation occurs in one step.  

## 计算特征值

我们可以QR分解迭代来计算特征值。对于第k 迭代来说，我们先将A 经过QR分解，
$$
\bold A_k = \bold Q _k \bold R_k
$$
那么下一次迭代的A 即为
$$
\bold A_{k+1} = \bold R_k \bold Q_k
$$
这是因为
$$
\bold A_{k+1} = \bold R_k \bold Q_k = \bold Q^T_k \bold Q_k \bold R_k \bold Q_k = \bold Q_k^T \bold A_k \bold Q_k = \bold Q_k^{-1}\bold A_k \bold Q_k
$$
也就是A{k+1}和A{k}是相似的，那么它们有相同的特征值。经过多次迭代后，A{k}会收敛到对角矩阵，而对角线上的元素就是特征值。

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

特征值算出之后，要计算特征向量，对于3x3矩阵也是非常简单。特征值的sqrt 就是奇异值。而右奇异矩阵 Vt 就是特征向量组成的矩阵。剩下的左奇异矩阵U也能很快算出来。

本方法的缺点是，当

## 方法一的缺点

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



## vegafem库

又是我们超级无敌酷炫的vegafem库，它是先计算矩阵A的特征值和特征向量，然后就能很快计算出奇异值。而特征值的计算用到了QL算法和householder分解。地址在https://github.com/starseeker/VegaFEM/blob/6252395422f96695e5403adaedd104040d7a7679/libraries/minivector/mat3d.cpp#L79。

vegafem库还处理了当sigma太小的情况。详细请看代码。

## irving04库

先求特征值，再求奇异值，也是可逆有限元 G. Irving, J. Teran, and R. Fedkiw. Invertible Finite Elements for Robust Simulation of Large Deformation 所使用的算法。

## pielet库

pielet

## IPC库

# 方法二：Eigen

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



这篇论文中的算法非常经典。

首先是这篇论文自己就开源了代码，可见http://pages.cs.wisc.edu/~sifakis/project_pages/svd.html。

AMD公司的开源库GPUopen中的FEMFX也实现了这里的代码，即https://github.com/GPUOpen-Effects/FEMFX/blob/master/amd_femfx/src/Common/FEMFXSvd3x3.cpp

Descent Methods for Elastic Body Simulation on the GPU 也开源了，可见https://web.cse.ohio-state.edu/~wang.3602/publications.html

另一个是个人开源的地址为https://github.com/benjones/quatSVD/blob/master/quatSVD.hpp

https://github.com/wi-re/tbtSVD

# 方法四


$$
\bold A = \begin{bmatrix} 3 & 4 & 2 \\ 4 & 5 & 2 \\ 2 & 5 & 3\end{bmatrix}
$$
首先Bi



Implicit-shifted Symmetric QR Singular Value Decomposition of 3 × 3 Matrices 

中介绍的算法也很受欢迎。

用于实现物质点法的的宾夕法尼亚大学开源库ziran中就有这篇论文自己的开源代码，地址在https://github.com/penn-graphics-research/ziran2019/blob/master/Lib/Ziran/Math/Linear/ImplicitQRSVD.h。

斯坦福大学的开源库physbam也使用了这种方法，地址在https://github.com/cuix-github/PhysBAM/blob/521b27e299f8add7ec42bb128f9cf9add4812eaf/PhysBAM_Tools/Matrices/MATRIX_3X3.cpp#L50

bullet3库也是如此https://github.com/romanpunia/tomahawk/blob/373f080ff8552dc770885f69a67f89edb6c98e09/src/supplies/bullet3/LinearMath/btImplicitQRSVD.h

最后是使用unity计算着色器实现的https://github.com/vanish87/UnitySVDComputeShader



## WILKINSON



https://dspace.mit.edu/bitstream/handle/1721.1/75282/18-335j-fall-2006/contents/lecture-notes/lec16.pdf





![image-20220114163126816](E:\mycode\UnityPhysicsPlayground\SingularValueDecomposition\image-20220114163126816.png)

```
import numpy as np
n = 3
A = np.array([[3,1,2],[2,7,2],[2,1,3]])
A = np.dot(A.T,A)

def wilksonShiftQR(A, iterations=100):
    Ak = np.copy(A)
    n = Ak.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        b11 = A[n-2,n-2]
        b22 = A[n-1,n-1]
        b12 = A[n-2,n-1]
        b21 = A[n-1,n-2]
        delta = (b11 - b22) / 2
        sign = 0
        if abs(delta) < 1e-10:
            sign = - 1
        else:
            sign = np.sign(delta)
        shift = b22 - (sign*b21*b21) / (abs(delta) + np.sqrt(delta*delta + b21*b21))
        smult = shift * np.eye(n)
        Q, R = np.linalg.qr(Ak - smult)
        Ak = np.dot(R,Q) + smult
    return Ak

eiv0 =  wilksonShiftQR(A)
eiv1 = np.linalg.eig(A)[0]
```

## 主循环

$$
\bold B = \begin{bmatrix}\alpha_1 & \beta_1 & 0  \\ 0 & \alpha_2 & \beta_2 \\ 0 & 0 & \alpha_3 \end{bmatrix}
$$

那么
$$
\bold T = \bold B^T \bold B = \begin{bmatrix} \alpha_1^2 & \alpha_1 \beta_1 & 0 \\ \alpha_1 \beta_1 & \alpha_2^2 + \beta_1^2 & \alpha_2 \beta_2 \\ 0 & \alpha_2 \beta_2 & \alpha_3^2 + \beta_2^2 \end{bmatrix}
$$


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
