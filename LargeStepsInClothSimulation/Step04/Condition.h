#pragma once
#include <Eigen\Core>
#include <Eigen\Dense>
using namespace Eigen;



const int Nx = 2;
const int Ny = 2;
const int node_num = Nx * Ny;

#ifdef SPARSE_DF
Matrix<float, node_num * 3, 7 * 3> dfdx;
Matrix<float, node_num * 3, 7 * 3> dfdv;
#else
Matrix<float, node_num * 3, node_num * 3> dfdx;
Matrix<float, node_num * 3, node_num * 3> dfdv;
#endif
VectorXf forces(node_num * 3);

class Condition
{
public:
	float alpha_shear;
	float k_shear;
	float damping_shear;

	float alpha_stretch;
	float k_stretch;
	float damping_stretch;

	Vector3f v01, v02, v32, v31, v21;
	Vector3f e01, e02, e32, e31, e21;
	float c00, c01, c02, c13, c11, c12;
	Vector3f n0, n1;
	Vector3f b00, b01, b02, b13, b11, b12;
	float d00, d01, d02, d13, d11, d12;
	Matrix3f identity;

	Condition()
	{
		v01 = Vector3f();

		alpha_shear = 1;
		k_shear = 1;
		damping_shear = 2;

		alpha_stretch = 1;
		k_stretch = 1;
		damping_stretch = 2;

		identity << 1, 0, 0, 0, 1, 0, 0, 0, 1;
	}



public:

	int Relative(int idx0, int idx1)
	{
		/*
				5-----6
				|       |
		3 --- 0 ----4
		|       |
		1 --- 2

		*/
		switch (idx1 - idx0)
		{
		case 0:
			return 0;
		case 1:
			return 4;
		case Nx:
			return 5;
		case Nx + 1:
			return 6;
		case -1:
			return 3;
		case -Nx:
			return 2;
		case -Nx - 1:
			return 1;
		default:
			break;
		}
		return 0;
	}


	void ComputeShearForces(Vector3f wu, Vector3f wv, Vector3f dwu, Vector3f dwv, Vector3f p0, Vector3f p1, Vector3f p2, Vector3f v0, Vector3f v1, Vector3f v2, int idx0, int idx1, int idx2)//, MatrixXf dfdx)
	{

		float ShearCondition = alpha_shear * wu.dot(wv);


		Vector3f dcdx0 = alpha_shear * (dwu(0) * wv + dwv(0) * wu);
		Vector3f dcdx1 = alpha_shear * (dwu(1) * wv + dwv(1) * wu);
		Vector3f dcdx2 = alpha_shear * (dwu(2) * wv + dwv(2) * wu);

		Matrix3f d2dx0x0 = alpha_shear * (dwu(0) * dwv(0) + dwu(0) * dwu(0)) * identity;
		Matrix3f d2dx0x1 = alpha_shear * (dwu(0) * dwv(1) + dwu(1) * dwu(0)) * identity;
		Matrix3f d2dx0x2 = alpha_shear * (dwu(0) * dwv(2) + dwu(2) * dwu(0)) * identity;

		Matrix3f d2dx1x0 = alpha_shear * (dwu(1) * dwv(0) + dwu(0) * dwu(1)) * identity;
		Matrix3f d2dx1x1 = alpha_shear * (dwu(1) * dwv(1) + dwu(1) * dwu(1)) * identity;
		Matrix3f d2dx1x2 = alpha_shear * (dwu(1) * dwv(2) + dwu(2) * dwu(1)) * identity;

		Matrix3f d2dx2x0 = alpha_shear * (dwu(2) * dwv(1) + dwu(1) * dwu(2)) * identity;
		Matrix3f d2dx2x1 = alpha_shear * (dwu(2) * dwv(1) + dwu(1) * dwu(2)) * identity;
		Matrix3f d2dx2x2 = alpha_shear * (dwu(2) * dwv(2) + dwu(2) * dwu(2)) * identity;

		// 计算力

		forces.segment<3>(idx0 * 3) += -k_shear * ShearCondition * dcdx0;
		forces.segment<3>(idx1 * 3) += -k_shear * ShearCondition * dcdx1;
		forces.segment<3>(idx2 * 3) += -k_shear * ShearCondition * dcdx2;

		
		float dcdt = dcdx0.dot(v0) + dcdx1.dot(v1) + dcdx2.dot(v2);

		forces.segment<3>(idx0 * 3) += -damping_shear * dcdt * dcdx0;
		forces.segment<3>(idx1 * 3) += -damping_shear * dcdt * dcdx1;
		forces.segment<3>(idx2 * 3) += -damping_shear * dcdt * dcdx2;

		
		Matrix3f df0dx0 = -k_shear * (dcdx0 * dcdx0.transpose() + ShearCondition * d2dx0x0);
		Matrix3f df0dx1 = -k_shear * (dcdx0 * dcdx1.transpose() + ShearCondition * d2dx0x1);
		Matrix3f df0dx2 = -k_shear * (dcdx0 * dcdx2.transpose() + ShearCondition * d2dx0x2);

		
		Matrix3f df1dx0 = -k_shear * (dcdx1 * dcdx0.transpose() + ShearCondition * d2dx1x0);
		Matrix3f df1dx1 = -k_shear * (dcdx1 * dcdx1.transpose() + ShearCondition * d2dx1x1);
		Matrix3f df1dx2 = -k_shear * (dcdx1 * dcdx2.transpose() + ShearCondition * d2dx1x2);

		Matrix3f df2dx0 = -k_shear * (dcdx2 * dcdx0.transpose() + ShearCondition * d2dx2x0);
		Matrix3f df2dx1 = -k_shear * (dcdx2 * dcdx1.transpose() + ShearCondition * d2dx2x1);
		Matrix3f df2dx2 = -k_shear * (dcdx2 * dcdx2.transpose() + ShearCondition * d2dx2x2);


		// 先搞自己的
#ifdef SPARSE_DF
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx0) * 3) += df0dx0;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx1) * 3) += df0dx1;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx2) * 3) += df0dx2;

		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx0) * 3) += df1dx0;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx1) * 3) += df1dx1;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx2) * 3) += df1dx2;

		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx0) * 3) += df2dx0;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx1) * 3) += df2dx1;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx2) * 3) += df2dx2;
#else
		dfdx.block<3, 3>(idx0 * 3, idx0 * 3) += df0dx0;
		dfdx.block<3, 3>(idx0 * 3, idx1 * 3) += df0dx1;
		dfdx.block<3, 3>(idx0 * 3, idx2 * 3) += df0dx2;

		dfdx.block<3, 3>(idx1 * 3, idx0 * 3) += df1dx0;
		dfdx.block<3, 3>(idx1 * 3, idx1 * 3) += df1dx1;
		dfdx.block<3, 3>(idx1 * 3, idx2 * 3) += df1dx2;

		dfdx.block<3, 3>(idx2 * 3, idx0 * 3) += df2dx0;
		dfdx.block<3, 3>(idx2 * 3, idx1 * 3) += df2dx1;
		dfdx.block<3, 3>(idx2 * 3, idx2 * 3) += df2dx2;
#endif


		Matrix3f df0dv0 = -damping_shear * (dcdx0 * dcdx0.transpose());
		Matrix3f df0dv1 = -damping_shear * (dcdx0 * dcdx1.transpose());
		Matrix3f df0dv2 = -damping_shear * (dcdx0 * dcdx2.transpose());

		Matrix3f df1dv0 = -damping_shear * (dcdx1 * dcdx0.transpose());
		Matrix3f df1dv1 = -damping_shear * (dcdx1 * dcdx1.transpose());
		Matrix3f df1dv2 = -damping_shear * (dcdx1 * dcdx2.transpose());

		Matrix3f df2dv0 = -damping_shear * (dcdx2 * dcdx0.transpose());
		Matrix3f df2dv1 = -damping_shear * (dcdx2 * dcdx1.transpose());
		Matrix3f df2dv2 = -damping_shear * (dcdx2 * dcdx2.transpose());

#ifdef SPARSE_DF
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx0) * 3) += df0dv0;
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx1) * 3) += df0dv1;
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx2) * 3) += df0dv2;

		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx0) * 3) += df1dv0;
		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx1) * 3) += df1dv1;
		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx2) * 3) += df1dv2;
	
		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx0) * 3) += df2dv0;
		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx1) * 3) += df2dv1;
		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx2) * 3) += df2dv2;
#else
		dfdv.block<3, 3>(idx0 * 3, idx0 * 3) += df0dv0;
		dfdv.block<3, 3>(idx0 * 3, idx1 * 3) += df0dv1;
		dfdv.block<3, 3>(idx0 * 3, idx2 * 3) += df0dv2;

		dfdv.block<3, 3>(idx1 * 3, idx0 * 3) += df1dv0;
		dfdv.block<3, 3>(idx1 * 3, idx1 * 3) += df1dv1;
		dfdv.block<3, 3>(idx1 * 3, idx2 * 3) += df1dv2;

		dfdv.block<3, 3>(idx2 * 3, idx0 * 3) += df2dv0;
		dfdv.block<3, 3>(idx2 * 3, idx1 * 3) += df2dv1;
		dfdv.block<3, 3>(idx2 * 3, idx2 * 3) += df2dv2;
#endif



	}
	void ComputeStretchForces(Vector3f wu, Vector3f wv, Vector3f dwu, Vector3f dwv, Vector3f p0, Vector3f p1, Vector3f p2, Vector3f v0, Vector3f v1, Vector3f v2, int idx0, int idx1, int idx2)//, MatrixXf dfdx)
	{

		float wuNorm = wu.norm();
		float wvNorm = wv.norm();

		float cu = alpha_stretch * (wuNorm - 1);
		float cv = alpha_stretch * (wvNorm - 1);

		Vector3f dcudx0 = alpha_stretch * dwu(0) * wu / wuNorm;
		Vector3f dcudx1 = alpha_stretch * dwu(1) * wu / wuNorm;
		Vector3f dcudx2 = alpha_stretch * dwu(2) * wu / wuNorm;

		Vector3f dcvdx0 = alpha_stretch * dwv(0) * wv / wvNorm;
		Vector3f dcvdx1 = alpha_stretch * dwv(1) * wv / wvNorm;
		Vector3f dcvdx2 = alpha_stretch * dwv(2) * wv / wvNorm;

		forces.segment<3>(idx0 * 3) += -k_stretch * (cu * dcudx0 + cv * dcvdx0);
		forces.segment<3>(idx1 * 3) += -k_stretch * (cu * dcudx1 + cv * dcvdx1);
		forces.segment<3>(idx2 * 3) += -k_stretch * (cu * dcudx2 + cv * dcvdx2);
	
		

		Matrix3f wuMatrix = (Matrix3f::Identity() - wu * wu.transpose() / (wuNorm * wuNorm));
		Matrix3f wvMatrix = (Matrix3f::Identity() - wv * wv.transpose() / (wvNorm * wvNorm));

		Matrix3f d2cudx0x0 = (alpha_stretch / wuNorm) * dwu(0) * dwu(0) * wuMatrix;
		Matrix3f d2cudx0x1 = (alpha_stretch / wuNorm) * dwu(0) * dwu(1) * wuMatrix;
		Matrix3f d2cudx0x2 = (alpha_stretch / wuNorm) * dwu(0) * dwu(2) * wuMatrix;

		Matrix3f d2cudx1x0 = (alpha_stretch / wuNorm) * dwu(1) * dwu(0) * wuMatrix;
		Matrix3f d2cudx1x1 = (alpha_stretch / wuNorm) * dwu(1) * dwu(1) * wuMatrix;
		Matrix3f d2cudx1x2 = (alpha_stretch / wuNorm) * dwu(1) * dwu(2) * wuMatrix;

		Matrix3f d2cudx2x0 = (alpha_stretch / wuNorm) * dwu(2) * dwu(0) * wuMatrix;
		Matrix3f d2cudx2x1 = (alpha_stretch / wuNorm) * dwu(2) * dwu(1) * wuMatrix;
		Matrix3f d2cudx2x2 = (alpha_stretch / wuNorm) * dwu(2) * dwu(2) * wuMatrix;

		Matrix3f d2cvdx0x0 = (alpha_stretch / wvNorm) * dwv(0) * dwv(0) * wvMatrix;
		Matrix3f d2cvdx0x1 = (alpha_stretch / wvNorm) * dwv(0) * dwv(1) * wvMatrix;
		Matrix3f d2cvdx0x2 = (alpha_stretch / wvNorm) * dwv(0) * dwv(2) * wvMatrix;

		Matrix3f d2cvdx1x0 = (alpha_stretch / wvNorm) * dwv(1) * dwv(0) * wvMatrix;
		Matrix3f d2cvdx1x1 = (alpha_stretch / wvNorm) * dwv(1) * dwv(1) * wvMatrix;
		Matrix3f d2cvdx1x2 = (alpha_stretch / wvNorm) * dwv(1) * dwv(2) * wvMatrix;

		Matrix3f d2cvdx2x0 = (alpha_stretch / wvNorm) * dwv(2) * dwv(0) * wvMatrix;
		Matrix3f d2cvdx2x1 = (alpha_stretch / wvNorm) * dwv(2) * dwv(1) * wvMatrix;
		Matrix3f d2cvdx2x2 = (alpha_stretch / wvNorm) * dwv(2) * dwv(2) * wvMatrix;

		Matrix3f df0dx0 = -k_stretch * (dcudx0 * dcudx0.transpose() + cu * d2cudx0x0 + dcvdx0 * dcvdx0.transpose() + cv * d2cvdx0x0);
		Matrix3f df0dx1 = -k_stretch * (dcudx0 * dcudx1.transpose() + cu * d2cudx0x1 + dcvdx0 * dcvdx1.transpose() + cv * d2cvdx0x1);
		Matrix3f df0dx2 = -k_stretch * (dcudx0 * dcudx2.transpose() + cu * d2cudx0x2 + dcvdx0 * dcvdx2.transpose() + cv * d2cvdx0x2);

		Matrix3f df1dx0 = -k_stretch * (dcudx1 * dcudx0.transpose() + cu * d2cudx1x0 + dcvdx1 * dcvdx0.transpose() + cv * d2cvdx1x0);
		Matrix3f df1dx1 = -k_stretch * (dcudx1 * dcudx1.transpose() + cu * d2cudx1x1 + dcvdx1 * dcvdx1.transpose() + cv * d2cvdx1x1);
		Matrix3f df1dx2 = -k_stretch * (dcudx1 * dcudx2.transpose() + cu * d2cudx1x2 + dcvdx1 * dcvdx2.transpose() + cv * d2cvdx1x2);

		Matrix3f df2dx0 = -k_stretch * (dcudx2 * dcudx0.transpose() + cu * d2cudx2x0 + dcvdx2 * dcvdx0.transpose() + cv * d2cvdx2x0);
		Matrix3f df2dx1 = -k_stretch * (dcudx2 * dcudx1.transpose() + cu * d2cudx2x1 + dcvdx2 * dcvdx1.transpose() + cv * d2cvdx2x1);
		Matrix3f df2dx2 = -k_stretch * (dcudx2 * dcudx2.transpose() + cu * d2cudx2x2 + dcvdx2 * dcvdx2.transpose() + cv * d2cvdx2x2);

#ifdef SPARSE_DF
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx0) * 3) += df0dx0;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx1) * 3) += df0dx1;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx2) * 3) += df0dx2;

		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx0) * 3) += df1dx0;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx1) * 3) += df1dx1;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx2) * 3) += df1dx2;

		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx0) * 3) += df2dx0;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx1) * 3) += df2dx1;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx2) * 3) += df2dx2;
#else
		dfdx.block<3, 3>(idx0 * 3, idx0 * 3) += df0dx0;
		dfdx.block<3, 3>(idx0 * 3, idx1 * 3) += df0dx1;
		dfdx.block<3, 3>(idx0 * 3, idx2 * 3) += df0dx2;

		dfdx.block<3, 3>(idx1 * 3, idx0 * 3) += df1dx0;
		dfdx.block<3, 3>(idx1 * 3, idx1 * 3) += df1dx1;
		dfdx.block<3, 3>(idx1 * 3, idx2 * 3) += df1dx2;

		dfdx.block<3, 3>(idx2 * 3, idx0 * 3) += df2dx0;
		dfdx.block<3, 3>(idx2 * 3, idx1 * 3) += df2dx1;
		dfdx.block<3, 3>(idx2 * 3, idx2 * 3) += df2dx2;
#endif

		// 速度阻尼项
		float dcudt = dcudx0.dot(v0) + dcudx1.dot(v1) + dcudx2.dot(v2);
		float dcvdt = dcvdx0.dot(v0) + dcvdx1.dot(v1) + dcvdx2.dot(v2);

		forces.segment<3>(idx0 * 3) += -damping_stretch * (dcudt * dcudx0 + dcvdt * dcvdx0);
		forces.segment<3>(idx1 * 3) += -damping_stretch * (dcudt * dcudx1 + dcvdt * dcvdx1);
		forces.segment<3>(idx2 * 3) += -damping_stretch * (dcudt * dcudx2 + dcvdt * dcvdx2);
	
		Matrix3f df0dv0 = -damping_stretch * (dcudx0 * dcudx0.transpose() + dcvdx0 * dcvdx0.transpose());
		Matrix3f df0dv1 = -damping_stretch * (dcudx0 * dcudx1.transpose() + dcvdx0 * dcvdx1.transpose());
		Matrix3f df0dv2 = -damping_stretch * (dcudx0 * dcudx2.transpose() + dcvdx0 * dcvdx2.transpose());

		Matrix3f df1dv0 = -damping_stretch * (dcudx1 * dcudx0.transpose() + dcvdx1 * dcvdx0.transpose());
		Matrix3f df1dv1 = -damping_stretch * (dcudx1 * dcudx1.transpose() + dcvdx1 * dcvdx1.transpose());
		Matrix3f df1dv2 = -damping_stretch * (dcudx1 * dcudx2.transpose() + dcvdx1 * dcvdx2.transpose());

		Matrix3f df2dv0 = -damping_stretch * (dcudx2 * dcudx0.transpose() + dcvdx2 * dcvdx0.transpose());
		Matrix3f df2dv1 = -damping_stretch * (dcudx2 * dcudx1.transpose() + dcvdx2 * dcvdx1.transpose());
		Matrix3f df2dv2 = -damping_stretch * (dcudx2 * dcudx2.transpose() + dcvdx2 * dcvdx2.transpose());

#ifdef SPARSE_DF
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx0) * 3) += df0dv0;
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx1) * 3) += df0dv1;
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx2) * 3) += df0dv2;

		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx0) * 3) += df1dv0;
		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx1) * 3) += df1dv1;
		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx2) * 3) += df1dv2;

		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx0) * 3) += df2dv0;
		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx1) * 3) += df2dv1;
		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx2) * 3) += df2dv2;
#else
		dfdv.block<3, 3>(idx0 * 3, idx0 * 3) += df0dv0;
		dfdv.block<3, 3>(idx0 * 3, idx1 * 3) += df0dv1;
		dfdv.block<3, 3>(idx0 * 3, idx2 * 3) += df0dv2;

		dfdv.block<3, 3>(idx1 * 3, idx0 * 3) += df1dv0;
		dfdv.block<3, 3>(idx1 * 3, idx1 * 3) += df1dv1;
		dfdv.block<3, 3>(idx1 * 3, idx2 * 3) += df1dv2;

		dfdv.block<3, 3>(idx2 * 3, idx0 * 3) += df2dv0;
		dfdv.block<3, 3>(idx2 * 3, idx1 * 3) += df2dv1;
		dfdv.block<3, 3>(idx2 * 3, idx2 * 3) += df2dv2;
#endif
	
	}
	void computeBendingForce(Vector3f p0, Vector3f p1, Vector3f p2, Vector3f p3,
		Vector3f v0, Vector3f v1, Vector3f v2, Vector3f v3)
	{
		/*
			// triangles are laid out like this:
			//     0
			//    / \
			//   / 0 \
			//  1-----2
			//   \ 1 /
			//    \ /
			//     3

		//2  --- 3
		// |   \   |
		// 0 --- 1
		v01 = p1 - p0;
		v02 = p2 - p0;
		v32 = p2 - p3;
		v31 = p1 - p3;
		v21 = p1 - p2;

		// normalized edge vectors:
		e01 = v01.normalized();
		e02 = v02.normalized();
		e32 = v32.normalized();
		e31 = v31.normalized();
		e21 = v21.normalized();

		// cosines of the angles at each of the points for each of the triangles:
		c00 = e01.dot(e02);
		c01 = e01.dot(e21);
		c02 = -e02.dot(e21);

		c13 = e31.dot(e32);
		c11 = e21.dot(e31);
		c12 = -e32.dot(e21);

		// normalized triangle normals:
		n0 = (e21.cross(e01)).normalized();
		n1 = (e32.cross(e21)).normalized();

		// normalized binormals for each triangle, pointing from a vertex to its opposite edge:

		float res =  e21.dot(e01);
		Vector3f res0 = e21 * e21.dot(e01);

		b00 = (e01 - e21 * (e21.dot(e01))).normalized();
		b01 = (-e01 - e02 * (e02.dot(-e01))).normalized();
		b02 = (-e02 - e01 * (e01.dot(-e02))).normalized();

		b13 = (e32 - e21 * (e21.dot(e32))).normalized();
		b12 = (-e32 - e31 * (e31.dot(-e32))).normalized();
		b11 = (-e31 - e32 * (e32.dot(-e31))).normalized();

		// vertex distances to opposite edges:
		d00 = (v01 - v01.dot(v21) / v21.dot(v21) * v21).norm();//b00.dot(v01);
		d01 = b01.dot(-v21);
		d02 = b02.dot(-v02);

		d11 = b11.dot(-v31);
		d12 = b12.dot(v21);
		d13 = b13.dot(v31);

		Matrix3f dn0dP0 = b00 * n0.transpose() / d00;

		float t = 1;
		*/
	}



};