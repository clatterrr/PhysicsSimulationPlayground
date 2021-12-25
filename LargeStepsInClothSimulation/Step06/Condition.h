#pragma once
#include <Eigen\Core>
#include <Eigen\Dense>
using namespace Eigen;

#define SPARSE_DF

const int Nx = 2;
const int Ny = 2;
const int node_num = Nx * Ny;

#ifdef SPARSE_DF
Matrix<float, node_num * 3, 9 * 3> dfdx;
Matrix<float, node_num * 3, 9 * 3> dfdv;
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

	Vector3f v10, v20, v23, v13, v12;
	Vector3f e10, e20, e23, e13, e12;
	float c00, c01, c02, c13, c11, c12;
	Vector3f n0, n1;
	Vector3f b00, b01, b02, b13, b11, b12;
	float d00, d01, d02, d13, d11, d12;
	Matrix3f identity;

	// Bend
	float alpha_bend;
	float k_bend;
	float damping_bend;

	Vector3f dThetadx0, dThetadx1, dThetadx2, dThetadx3;

	float theta, sinTheta, cosTheta, dThetadt;

	Matrix3f dn0dx0, dn0dx1, dn0dx2, dn0dx3, dn1dx0, dn1dx1, dn1dx2, dn1dx3;
	// derivatives of the cosines:
	Vector3f dc01dx0, dc01dx1, dc01dx2, dc01dx3, dc02dx0, dc02dx1, dc02dx2, dc02dx3;
	Vector3f dc11dx0, dc11dx1, dc11dx2, dc11dx3, dc12dx0, dc12dx1, dc12dx2, dc12dx3;
	// derivatives of the perpendicular distances:
	Vector3f dd00dx0, dd00dx1, dd00dx2, dd00dx3;
	Vector3f dd01dx0, dd01dx1, dd01dx2, dd01dx3;
	Vector3f dd02dx0, dd02dx1, dd02dx2, dd02dx3;
	Vector3f dd11dx0, dd11dx1, dd11dx2, dd11dx3;
	Vector3f dd12dx0, dd12dx1, dd12dx2, dd12dx3;
	Vector3f dd13dx0, dd13dx1, dd13dx2, dd13dx3;
	// second derivatives of theta with respect to the different vertex positions:
	Matrix3f d2Thetadx0dx0, d2Thetadx0dx1, d2Thetadx0dx2, d2Thetadx0dx3;
	Matrix3f d2Thetadx1dx0, d2Thetadx1dx1, d2Thetadx1dx2, d2Thetadx1dx3;
	Matrix3f d2Thetadx2dx0, d2Thetadx2dx1, d2Thetadx2dx2, d2Thetadx2dx3;
	Matrix3f d2Thetadx3dx0, d2Thetadx3dx1, d2Thetadx3dx2, d2Thetadx3dx3;

	// 计算力用到的
	Matrix3f df0dx0, df0dx1, df0dx2, df0dx3;
	Matrix3f df1dx0, df1dx1, df1dx2, df1dx3;
	Matrix3f df2dx0, df2dx1, df2dx2, df2dx3;
	Matrix3f df3dx0, df3dx1, df3dx2, df3dx3;

	Matrix3f df0dv0, df0dv1, df0dv2, df0dv3;
	Matrix3f df1dv0, df1dv1, df1dv2, df1dv3;
	Matrix3f df2dv0, df2dv1, df2dv2, df2dv3;
	Matrix3f df3dv0, df3dv1, df3dv2, df3dv3;

	Condition()
	{

		alpha_shear = 1;
		k_shear = 1;
		damping_shear = 2;

		alpha_stretch = 1;
		k_stretch = 1;
		damping_stretch = 2;

		alpha_bend = 1;
		k_bend = 1;
		damping_bend = 2;

		identity << 1, 0, 0, 0, 1, 0, 0, 0, 1;
	}



public:

	int Relative(int idx0, int idx1)
	{

		/*
		6---- 7-----8
		|		|       |
		4 --- 0 ----5
		|       |       |
		1 --- 2 ----3

		*/
		int x0 = idx1 % Nx - idx0 % Nx;
		int y0 = (int)(idx1 / Nx) - (int)(idx0 / Nx);
		if (x0 == 0 && y0 == 0)return 0;
		else if (x0 == -1 && y0 == -1)return 1;
		else if (x0 == 0 && y0 == -1)return 2;
		else if (x0 == 1 && y0 == -1)return 3;
		else if (x0 == -1 && y0 == 0)return 4;
		else if (x0 == 1 && y0 == 0)return 5;
		else if (x0 == -1 && y0 == 1)return 6;
		else if (x0 == 0 && y0 == 1)return 7;
		else if (x0 == 1 && y0 == 1)return 8;
		return -1;
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


		df0dx0 = -k_shear * (dcdx0 * dcdx0.transpose() + ShearCondition * d2dx0x0);
		df0dx1 = -k_shear * (dcdx0 * dcdx1.transpose() + ShearCondition * d2dx0x1);
		df0dx2 = -k_shear * (dcdx0 * dcdx2.transpose() + ShearCondition * d2dx0x2);


		df1dx0 = -k_shear * (dcdx1 * dcdx0.transpose() + ShearCondition * d2dx1x0);
		df1dx1 = -k_shear * (dcdx1 * dcdx1.transpose() + ShearCondition * d2dx1x1);
		df1dx2 = -k_shear * (dcdx1 * dcdx2.transpose() + ShearCondition * d2dx1x2);

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


		df0dv0 = -damping_shear * (dcdx0 * dcdx0.transpose());
		df0dv1 = -damping_shear * (dcdx0 * dcdx1.transpose());
		df0dv2 = -damping_shear * (dcdx0 * dcdx2.transpose());

		df1dv0 = -damping_shear * (dcdx1 * dcdx0.transpose());
		df1dv1 = -damping_shear * (dcdx1 * dcdx1.transpose());
		df1dv2 = -damping_shear * (dcdx1 * dcdx2.transpose());

		df2dv0 = -damping_shear * (dcdx2 * dcdx0.transpose());
		df2dv1 = -damping_shear * (dcdx2 * dcdx1.transpose());
		df2dv2 = -damping_shear * (dcdx2 * dcdx2.transpose());

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

		df0dx0 = -damping_shear * (d2dx0x0 * dcdt);
		df0dx1 = -damping_shear * (d2dx0x1 * dcdt);
		df0dx2 = -damping_shear * (d2dx0x2 * dcdt);


		df1dx0 = -damping_shear * (d2dx1x0 * dcdt);
		df1dx1 = -damping_shear * (d2dx1x1 * dcdt);
		df1dx2 = -damping_shear * (d2dx1x2 * dcdt);

		df2dx0 = -damping_shear * (d2dx2x0 * dcdt);
		df2dx1 = -damping_shear * (d2dx2x1 * dcdt);
		df2dx2 = -damping_shear * (d2dx2x2 * dcdt);

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

		df0dx0 = -k_stretch * (dcudx0 * dcudx0.transpose() + cu * d2cudx0x0 + dcvdx0 * dcvdx0.transpose() + cv * d2cvdx0x0);
		df0dx1 = -k_stretch * (dcudx0 * dcudx1.transpose() + cu * d2cudx0x1 + dcvdx0 * dcvdx1.transpose() + cv * d2cvdx0x1);
		df0dx2 = -k_stretch * (dcudx0 * dcudx2.transpose() + cu * d2cudx0x2 + dcvdx0 * dcvdx2.transpose() + cv * d2cvdx0x2);

		df1dx0 = -k_stretch * (dcudx1 * dcudx0.transpose() + cu * d2cudx1x0 + dcvdx1 * dcvdx0.transpose() + cv * d2cvdx1x0);
		df1dx1 = -k_stretch * (dcudx1 * dcudx1.transpose() + cu * d2cudx1x1 + dcvdx1 * dcvdx1.transpose() + cv * d2cvdx1x1);
		df1dx2 = -k_stretch * (dcudx1 * dcudx2.transpose() + cu * d2cudx1x2 + dcvdx1 * dcvdx2.transpose() + cv * d2cvdx1x2);

		df2dx0 = -k_stretch * (dcudx2 * dcudx0.transpose() + cu * d2cudx2x0 + dcvdx2 * dcvdx0.transpose() + cv * d2cvdx2x0);
		df2dx1 = -k_stretch * (dcudx2 * dcudx1.transpose() + cu * d2cudx2x1 + dcvdx2 * dcvdx1.transpose() + cv * d2cvdx2x1);
		df2dx2 = -k_stretch * (dcudx2 * dcudx2.transpose() + cu * d2cudx2x2 + dcvdx2 * dcvdx2.transpose() + cv * d2cvdx2x2);

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

		df0dv0 = -damping_stretch * (dcudx0 * dcudx0.transpose() + dcvdx0 * dcvdx0.transpose());
		df0dv1 = -damping_stretch * (dcudx0 * dcudx1.transpose() + dcvdx0 * dcvdx1.transpose());
		df0dv2 = -damping_stretch * (dcudx0 * dcudx2.transpose() + dcvdx0 * dcvdx2.transpose());

		df1dv0 = -damping_stretch * (dcudx1 * dcudx0.transpose() + dcvdx1 * dcvdx0.transpose());
		df1dv1 = -damping_stretch * (dcudx1 * dcudx1.transpose() + dcvdx1 * dcvdx1.transpose());
		df1dv2 = -damping_stretch * (dcudx1 * dcudx2.transpose() + dcvdx1 * dcvdx2.transpose());

		df2dv0 = -damping_stretch * (dcudx2 * dcudx0.transpose() + dcvdx2 * dcvdx0.transpose());
		df2dv1 = -damping_stretch * (dcudx2 * dcudx1.transpose() + dcvdx2 * dcvdx1.transpose());
		df2dv2 = -damping_stretch * (dcudx2 * dcudx2.transpose() + dcvdx2 * dcvdx2.transpose());

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

		df0dx0 = -damping_stretch * (d2cudx0x0 * dcudt + d2cvdx0x0 * dcvdt);
		df0dx1 = -damping_stretch * (d2cudx0x1 * dcudt + d2cvdx0x1 * dcvdt);
		df0dx2 = -damping_stretch * (d2cudx0x2 * dcudt + d2cvdx0x2 * dcvdt);


		df1dx0 = -damping_stretch * (d2cudx1x0 * dcudt + d2cvdx1x0 * dcvdt);
		df1dx1 = -damping_stretch * (d2cudx1x1 * dcudt + d2cvdx1x1 * dcvdt);
		df1dx2 = -damping_stretch * (d2cudx1x2 * dcudt + d2cvdx1x2 * dcvdt);

		df2dx0 = -damping_stretch * (d2cudx2x0 * dcudt + d2cvdx2x0 * dcvdt);
		df2dx1 = -damping_stretch * (d2cudx2x1 * dcudt + d2cvdx2x1 * dcvdt);
		df2dx2 = -damping_stretch * (d2cudx2x2 * dcudt + d2cvdx2x2 * dcvdt);

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

	}
	void computeBendingForce(Vector3f p0, Vector3f p1, Vector3f p2, Vector3f p3,
		Vector3f v0, Vector3f v1, Vector3f v2, Vector3f v3,
		int idx0, int idx1, int idx2, int idx3)
	{

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
		v10 = p1 - p0;
		v20 = p2 - p0;
		v23 = p2 - p3;
		v13 = p1 - p3;
		v12 = p1 - p2;

		// normalized edge vectors:
		e10 = v10.normalized();
		e20 = v20.normalized();
		e23 = v23.normalized();
		e13 = v13.normalized();
		e12 = v12.normalized();

		// cosines of the angles at each of the points for each of the triangles:
		c00 = e10.dot(e20);
		c01 = e10.dot(e12);
		c02 = -e20.dot(e12);

		c13 = e13.dot(e23);
		c11 = e12.dot(e13);
		c12 = -e23.dot(e12);

		// normalized triangle normals:
		n0 = (e12.cross(e10)).normalized();
		n1 = (e23.cross(e12)).normalized();

		// normalized binormals for each triangle, pointing from a vertex to its opposite edge:

		b00 = (e10 - e12 * (e12.dot(e10))).normalized();
		b01 = (-e10 - e20 * (e20.dot(-e10))).normalized();
		b02 = (-e20 - e10 * (e10.dot(-e20))).normalized();

		b13 = (e23 - e12 * (e12.dot(e23))).normalized();
		b12 = (-e23 - e13 * (e13.dot(-e23))).normalized();
		b11 = (-e13 - e23 * (e23.dot(-e13))).normalized();

		// vertex distances to opposite edges:
		d00 = b00.dot(v10);
		d01 = b01.dot(-v12);
		d02 = b02.dot(-v20);

		d11 = b11.dot(-v13);
		d12 = b12.dot(v12);
		d13 = b13.dot(v13);

		sinTheta = n1.dot(b00);
		cosTheta = n0.dot(n1);
		theta = atan2(sinTheta, cosTheta);

		// 法向量求导
		dn0dx0 = b00 * n0.transpose() / d00;
		dn0dx1 = b01 * n0.transpose() / d01;
		dn0dx2 = b02 * n0.transpose() / d02;
		dn0dx3 = Matrix3f::Zero();

		dn1dx0 = Matrix3f::Zero();
		dn1dx1 = b11 * n1.transpose() / d11;
		dn1dx2 = b12 * n1.transpose() / d12;
		dn1dx3 = b13 * n1.transpose() / d13;

		// 角度求导
		dThetadx0 = -n0 / d00;
		dThetadx1 = c02 * n0 / d01 + c12 * n1 / d11;
		dThetadx2 = c01 * n0 / d02 + c11 * n1 / d12;
		dThetadx3 = -n1 / d13;

		//距离求导
		dd00dx0 = -b00;
		dd00dx1 = b00 * -v12.dot(v20) / v12.dot(v12);
		dd00dx2 = b00 * v12.dot(v10) / v12.dot(v12);
		dd00dx3 = Vector3f::Zero();

		dd01dx0 = b01 * v20.dot(-v12) / v20.dot(v20);
		dd01dx1 = -b01;
		dd01dx2 = b01 * v20.dot(v10) / v20.dot(v20);
		dd01dx3 = Vector3f::Zero();

		dd02dx0 = b02 * v10.dot(v12) / v10.dot(v10);
		dd02dx1 = b02 * v10.dot(v20) / v10.dot(v10);
		dd02dx2 = -b02;
		dd02dx3 = Vector3f::Zero();

		dd11dx0 = Vector3f::Zero();
		dd11dx1 = -b11;
		dd11dx2 = b11 * v23.dot(v13) / v23.dot(v23);
		dd11dx3 = b11 * v23.dot(-v12) / v23.dot(v23);

		dd12dx0 = Vector3f::Zero();
		dd12dx1 = b12 * v13.dot(v23) / v13.dot(v13);
		dd12dx2 = -b12;
		dd12dx3 = b12 * v13.dot(v12) / v13.dot(v13);

		dd13dx0 = Vector3f::Zero();
		dd13dx1 = b13 * v12.dot(-v23) / v12.dot(v12);
		dd13dx2 = b13 * v12.dot(v13) / v12.dot(v12);
		dd13dx3 = -b13;

		//余弦求导
		dc01dx0 = -b02 * b00.dot(v10) / v10.dot(v10);
		dc01dx2 = -b00 * b02.dot(v12) / v12.dot(v12);
		dc01dx1 = -dc01dx0 - dc01dx2;
		dc01dx3 = Vector3f::Zero();

		dc02dx0 = -b01 * b00.dot(v20) / v20.dot(v20);
		dc02dx1 = b00 * b01.dot(v12) / v12.dot(v12);
		dc02dx2 = -dc02dx0 - dc02dx1;
		dc01dx3 = Vector3f::Zero();

		dc11dx0 = Vector3f::Zero();
		dc11dx2 = -b13 * b12.dot(v12) / v12.dot(v12);
		dc11dx3 = -b12 * b13.dot(v13) / v13.dot(v13);
		dc11dx1 = -dc11dx2 - dc11dx3;

		dc12dx0 = Vector3f::Zero();
		dc12dx1 = b13 * b11.dot(v12) / v12.dot(v12);
		dc12dx3 = -b11 * b13.dot(v23) / v23.dot(v23);
		dc12dx2 = -dc12dx1 - dc12dx3;

		// 角度二次导
		d2Thetadx0dx0 = -dn0dx0 / d00 + n0 * dd00dx0.transpose() / (d00 * d00);
		d2Thetadx0dx1 = -dn0dx1 / d00 + n0 * dd00dx1.transpose() / (d00 * d00);
		d2Thetadx0dx2 = -dn0dx2 / d00 + n0 * dd00dx2.transpose() / (d00 * d00);
		d2Thetadx0dx3 = -dn0dx3 / d00 + n0 * dd00dx3.transpose() / (d00 * d00);

		d2Thetadx1dx0 = ((c02 / d01) * dn0dx0 + n0 * (d01 * dc02dx0 - c02 * dd01dx0).transpose() / (d01 * d01)) + ((c12 / d11) * dn1dx0 + n1 * (d11 * dc12dx0 - c12 * dd11dx0).transpose() / (d11 * d11));
		d2Thetadx1dx1 = ((c02 / d01) * dn0dx1 + n0 * (d01 * dc02dx1 - c02 * dd01dx1).transpose() / (d01 * d01)) + ((c12 / d11) * dn1dx1 + n1 * (d11 * dc12dx1 - c12 * dd11dx1).transpose() / (d11 * d11));
		d2Thetadx1dx2 = ((c02 / d01) * dn0dx2 + n0 * (d01 * dc02dx2 - c02 * dd01dx2).transpose() / (d01 * d01)) + ((c12 / d11) * dn1dx2 + n1 * (d11 * dc12dx2 - c12 * dd11dx2).transpose() / (d11 * d11));
		d2Thetadx1dx3 = ((c02 / d01) * dn0dx3 + n0 * (d01 * dc02dx3 - c02 * dd01dx3).transpose() / (d01 * d01)) + ((c12 / d11) * dn1dx3 + n1 * (d11 * dc12dx3 - c12 * dd11dx3).transpose() / (d11 * d11));

		d2Thetadx2dx0 = ((c01 / d02) * dn0dx0 + n0 * (d02 * dc01dx0 - c01 * dd02dx0).transpose() / (d02 * d02)) + ((c11 / d12) * dn1dx0 + n1 * (d12 * dc11dx0 - c11 * dd12dx0).transpose() / (d12 * d12));
		d2Thetadx2dx1 = ((c01 / d02) * dn0dx1 + n0 * (d02 * dc01dx1 - c01 * dd02dx1).transpose() / (d02 * d02)) + ((c11 / d12) * dn1dx1 + n1 * (d12 * dc11dx1 - c11 * dd12dx1).transpose() / (d12 * d12));
		d2Thetadx2dx2 = ((c01 / d02) * dn0dx2 + n0 * (d02 * dc01dx2 - c01 * dd02dx2).transpose() / (d02 * d02)) + ((c11 / d12) * dn1dx2 + n1 * (d12 * dc11dx2 - c11 * dd12dx2).transpose() / (d12 * d12));
		d2Thetadx2dx3 = ((c01 / d02) * dn0dx3 + n0 * (d02 * dc01dx3 - c01 * dd02dx3).transpose() / (d02 * d02)) + ((c11 / d12) * dn1dx3 + n1 * (d12 * dc11dx3 - c11 * dd12dx3).transpose() / (d12 * d12));

		d2Thetadx3dx0 = -dn1dx0 / d13 + n1 * dd13dx0.transpose() / (d13 * d13);
		d2Thetadx3dx1 = -dn1dx1 / d13 + n1 * dd13dx1.transpose() / (d13 * d13);
		d2Thetadx3dx2 = -dn1dx2 / d13 + n1 * dd13dx2.transpose() / (d13 * d13);
		d2Thetadx3dx3 = -dn1dx3 / d13 + n1 * dd13dx3.transpose() / (d13 * d13);

		// 时间求导
		dThetadt = dThetadx0.dot(v0) + dThetadx1.dot(v1) + dThetadx2.dot(v2) + dThetadx3.dot(v3);

		// 计算力
		forces.segment<3>(idx0 * 3) -= k_bend * theta * dThetadx0;
		forces.segment<3>(idx1 * 3) -= k_bend * theta * dThetadx1;
		forces.segment<3>(idx2 * 3) -= k_bend * theta * dThetadx2;
		forces.segment<3>(idx3 * 3) -= k_bend * theta * dThetadx3;

		forces.segment<3>(idx0 * 3) -= damping_bend * dThetadt * dThetadx0;
		forces.segment<3>(idx1 * 3) -= damping_bend * dThetadt * dThetadx1;
		forces.segment<3>(idx2 * 3) -= damping_bend * dThetadt * dThetadx2;
		forces.segment<3>(idx3 * 3) -= damping_bend * dThetadt * dThetadx3;

		df0dx0 = -k_bend * (dThetadx0 * dThetadx0.transpose() + theta * d2Thetadx0dx0);
		df0dx1 = -k_bend * (dThetadx0 * dThetadx1.transpose() + theta * d2Thetadx0dx1);
		df0dx2 = -k_bend * (dThetadx0 * dThetadx2.transpose() + theta * d2Thetadx0dx2);
		df0dx3 = -k_bend * (dThetadx0 * dThetadx3.transpose() + theta * d2Thetadx0dx3);

		df1dx0 = -k_bend * (dThetadx1 * dThetadx0.transpose() + theta * d2Thetadx1dx0);
		df1dx1 = -k_bend * (dThetadx1 * dThetadx1.transpose() + theta * d2Thetadx1dx1);
		df1dx2 = -k_bend * (dThetadx1 * dThetadx2.transpose() + theta * d2Thetadx1dx2);
		df1dx3 = -k_bend * (dThetadx1 * dThetadx3.transpose() + theta * d2Thetadx1dx3);

		df2dx0 = -k_bend * (dThetadx2 * dThetadx0.transpose() + theta * d2Thetadx2dx0);
		df2dx1 = -k_bend * (dThetadx2 * dThetadx1.transpose() + theta * d2Thetadx2dx1);
		df2dx2 = -k_bend * (dThetadx2 * dThetadx2.transpose() + theta * d2Thetadx2dx2);
		df2dx3 = -k_bend * (dThetadx2 * dThetadx3.transpose() + theta * d2Thetadx2dx3);

		df3dx0 = -k_bend * (dThetadx3 * dThetadx0.transpose() + theta * d2Thetadx3dx0);
		df3dx1 = -k_bend * (dThetadx3 * dThetadx1.transpose() + theta * d2Thetadx3dx1);
		df3dx2 = -k_bend * (dThetadx3 * dThetadx2.transpose() + theta * d2Thetadx3dx2);
		df3dx3 = -k_bend * (dThetadx3 * dThetadx3.transpose() + theta * d2Thetadx3dx3);

#ifdef SPARSE_DF
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx0) * 3) += df0dx0;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx1) * 3) += df0dx1;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx2) * 3) += df0dx2;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx3) * 3) += df0dx3;

		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx0) * 3) += df1dx0;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx1) * 3) += df1dx1;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx2) * 3) += df1dx2;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx3) * 3) += df1dx3;

		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx0) * 3) += df2dx0;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx1) * 3) += df2dx1;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx2) * 3) += df2dx2;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx3) * 3) += df2dx3;

		dfdx.block<3, 3>(idx3 * 3, Relative(idx3, idx0) * 3) += df3dx0;
		dfdx.block<3, 3>(idx3 * 3, Relative(idx3, idx1) * 3) += df3dx1;
		dfdx.block<3, 3>(idx3 * 3, Relative(idx3, idx2) * 3) += df3dx2;
		dfdx.block<3, 3>(idx3 * 3, Relative(idx3, idx3) * 3) += df3dx3;
#else
		dfdx.block<3, 3>(idx0 * 3, idx0 * 3) += df0dx0;
		dfdx.block<3, 3>(idx0 * 3, idx1 * 3) += df0dx1;
		dfdx.block<3, 3>(idx0 * 3, idx2 * 3) += df0dx2;
		dfdx.block<3, 3>(idx0 * 3, idx3 * 3) += df0dx3;

		dfdx.block<3, 3>(idx1 * 3, idx0 * 3) += df1dx0;
		dfdx.block<3, 3>(idx1 * 3, idx1 * 3) += df1dx1;
		dfdx.block<3, 3>(idx1 * 3, idx2 * 3) += df1dx2;
		dfdx.block<3, 3>(idx1 * 3, idx3 * 3) += df1dx3;

		dfdx.block<3, 3>(idx2 * 3, idx0 * 3) += df2dx0;
		dfdx.block<3, 3>(idx2 * 3, idx1 * 3) += df2dx1;
		dfdx.block<3, 3>(idx2 * 3, idx2 * 3) += df2dx2;
		dfdx.block<3, 3>(idx2 * 3, idx3 * 3) += df2dx3;

		dfdx.block<3, 3>(idx3 * 3, idx0 * 3) += df3dx0;
		dfdx.block<3, 3>(idx3 * 3, idx1 * 3) += df3dx1;
		dfdx.block<3, 3>(idx3 * 3, idx2 * 3) += df3dx2;
		dfdx.block<3, 3>(idx3 * 3, idx3 * 3) += df3dx3;
#endif


		df0dx0 = -damping_bend * (d2Thetadx0dx0 * dThetadt);
		df0dx1 = -damping_bend * (d2Thetadx0dx1 * dThetadt);
		df0dx2 = -damping_bend * (d2Thetadx0dx2 * dThetadt);
		df0dx3 = -damping_bend * (d2Thetadx0dx3 * dThetadt);

		df1dx0 = -damping_bend * (d2Thetadx1dx0 * dThetadt);
		df1dx1 = -damping_bend * (d2Thetadx1dx1 * dThetadt);
		df1dx2 = -damping_bend * (d2Thetadx1dx2 * dThetadt);
		df1dx3 = -damping_bend * (d2Thetadx1dx3 * dThetadt);

		df2dx0 = -damping_bend * (d2Thetadx2dx0 * dThetadt);
		df2dx1 = -damping_bend * (d2Thetadx2dx1 * dThetadt);
		df2dx2 = -damping_bend * (d2Thetadx2dx2 * dThetadt);
		df2dx3 = -damping_bend * (d2Thetadx2dx3 * dThetadt);

		df3dx0 = -damping_bend * (d2Thetadx3dx0 * dThetadt);
		df3dx1 = -damping_bend * (d2Thetadx3dx1 * dThetadt);
		df3dx2 = -damping_bend * (d2Thetadx3dx2 * dThetadt);
		df3dx3 = -damping_bend * (d2Thetadx3dx3 * dThetadt);

#ifdef SPARSE_DF
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx0) * 3) += df0dx0;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx1) * 3) += df0dx1;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx2) * 3) += df0dx2;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx3) * 3) += df0dx3;

		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx0) * 3) += df1dx0;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx1) * 3) += df1dx1;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx2) * 3) += df1dx2;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx3) * 3) += df1dx3;

		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx0) * 3) += df2dx0;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx1) * 3) += df2dx1;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx2) * 3) += df2dx2;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx3) * 3) += df2dx3;

		dfdx.block<3, 3>(idx3 * 3, Relative(idx3, idx0) * 3) += df3dx0;
		dfdx.block<3, 3>(idx3 * 3, Relative(idx3, idx1) * 3) += df3dx1;
		dfdx.block<3, 3>(idx3 * 3, Relative(idx3, idx2) * 3) += df3dx2;
		dfdx.block<3, 3>(idx3 * 3, Relative(idx3, idx3) * 3) += df3dx3;
#else
		dfdx.block<3, 3>(idx0 * 3, idx0 * 3) += df0dx0;
		dfdx.block<3, 3>(idx0 * 3, idx1 * 3) += df0dx1;
		dfdx.block<3, 3>(idx0 * 3, idx2 * 3) += df0dx2;
		dfdx.block<3, 3>(idx0 * 3, idx3 * 3) += df0dx3;

		dfdx.block<3, 3>(idx1 * 3, idx0 * 3) += df1dx0;
		dfdx.block<3, 3>(idx1 * 3, idx1 * 3) += df1dx1;
		dfdx.block<3, 3>(idx1 * 3, idx2 * 3) += df1dx2;
		dfdx.block<3, 3>(idx1 * 3, idx3 * 3) += df1dx3;

		dfdx.block<3, 3>(idx2 * 3, idx0 * 3) += df2dx0;
		dfdx.block<3, 3>(idx2 * 3, idx1 * 3) += df2dx1;
		dfdx.block<3, 3>(idx2 * 3, idx2 * 3) += df2dx2;
		dfdx.block<3, 3>(idx2 * 3, idx3 * 3) += df2dx3;

		dfdx.block<3, 3>(idx3 * 3, idx0 * 3) += df3dx0;
		dfdx.block<3, 3>(idx3 * 3, idx1 * 3) += df3dx1;
		dfdx.block<3, 3>(idx3 * 3, idx2 * 3) += df3dx2;
		dfdx.block<3, 3>(idx3 * 3, idx3 * 3) += df3dx3;
#endif

		df0dv0 = -damping_bend * (dThetadx0 * dThetadx0.transpose());
		df0dv1 = -damping_bend * (dThetadx0 * dThetadx1.transpose());
		df0dv2 = -damping_bend * (dThetadx0 * dThetadx2.transpose());
		df0dv3 = -damping_bend * (dThetadx0 * dThetadx3.transpose());

		df1dv0 = -damping_bend * (dThetadx1 * dThetadx0.transpose());
		df1dv1 = -damping_bend * (dThetadx1 * dThetadx1.transpose());
		df1dv2 = -damping_bend * (dThetadx1 * dThetadx2.transpose());
		df1dv3 = -damping_bend * (dThetadx1 * dThetadx3.transpose());

		df2dv0 = -damping_bend * (dThetadx2 * dThetadx0.transpose());
		df2dv1 = -damping_bend * (dThetadx2 * dThetadx1.transpose());
		df2dv2 = -damping_bend * (dThetadx2 * dThetadx2.transpose());
		df2dv3 = -damping_bend * (dThetadx2 * dThetadx3.transpose());

		df3dv0 = -damping_bend * (dThetadx3 * dThetadx0.transpose());
		df3dv1 = -damping_bend * (dThetadx3 * dThetadx1.transpose());
		df3dv2 = -damping_bend * (dThetadx3 * dThetadx2.transpose());
		df3dv3 = -damping_bend * (dThetadx3 * dThetadx3.transpose());

#ifdef SPARSE_DF
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx0) * 3) += df0dv0;
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx1) * 3) += df0dv1;
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx2) * 3) += df0dv2;
		dfdv.block<3, 3>(idx0 * 3, Relative(idx0, idx3) * 3) += df0dv3;

		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx0) * 3) += df1dv0;
		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx1) * 3) += df1dv1;
		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx2) * 3) += df1dv2;
		dfdv.block<3, 3>(idx1 * 3, Relative(idx1, idx3) * 3) += df1dv3;

		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx0) * 3) += df2dv0;
		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx1) * 3) += df2dv1;
		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx2) * 3) += df2dv2;
		dfdv.block<3, 3>(idx2 * 3, Relative(idx2, idx3) * 3) += df2dv3;

		dfdv.block<3, 3>(idx3 * 3, Relative(idx3, idx0) * 3) += df3dv0;
		dfdv.block<3, 3>(idx3 * 3, Relative(idx3, idx1) * 3) += df3dv1;
		dfdv.block<3, 3>(idx3 * 3, Relative(idx3, idx2) * 3) += df3dv2;
		dfdv.block<3, 3>(idx3 * 3, Relative(idx3, idx3) * 3) += df3dv3;
#else
		dfdv.block<3, 3>(idx0 * 3, idx0 * 3) += df0dv0;
		dfdv.block<3, 3>(idx0 * 3, idx1 * 3) += df0dv1;
		dfdv.block<3, 3>(idx0 * 3, idx2 * 3) += df0dv2;
		dfdv.block<3, 3>(idx0 * 3, idx3 * 3) += df0dv3;

		dfdv.block<3, 3>(idx1 * 3, idx0 * 3) += df1dv0;
		dfdv.block<3, 3>(idx1 * 3, idx1 * 3) += df1dv1;
		dfdv.block<3, 3>(idx1 * 3, idx2 * 3) += df1dv2;
		dfdv.block<3, 3>(idx1 * 3, idx3 * 3) += df1dv3;

		dfdv.block<3, 3>(idx2 * 3, idx0 * 3) += df2dv0;
		dfdv.block<3, 3>(idx2 * 3, idx1 * 3) += df2dv1;
		dfdv.block<3, 3>(idx2 * 3, idx2 * 3) += df2dv2;
		dfdv.block<3, 3>(idx2 * 3, idx3 * 3) += df2dv3;

		dfdv.block<3, 3>(idx3 * 3, idx0 * 3) += df3dv0;
		dfdv.block<3, 3>(idx3 * 3, idx1 * 3) += df3dv1;
		dfdv.block<3, 3>(idx3 * 3, idx2 * 3) += df3dv2;
		dfdv.block<3, 3>(idx3 * 3, idx3 * 3) += df3dv3;
#endif
	}



};