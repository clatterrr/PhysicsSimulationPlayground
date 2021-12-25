#pragma once
#include <Eigen\Core>
#include <Eigen\Dense>
using namespace Eigen;


const int Nx = 2;
const int Ny = 2;
const int node_num = Nx * Ny;

Matrix<float, node_num * 3, 7 * 3> dfdx;
VectorXf forces(node_num * 3);

class Condition
{
public:
	float alpha_shear;
	float k_shear;
	float damping_shear;

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
		damping_shear = 10;

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

		/*
		Matrix3f df0dx0 = -k_shear * (dcdx0 * dcdx0.transpose() + ShearCondition * d2dx0x0);
		Matrix3f df0dx1 = -k_shear * (dcdx0 * dcdx1.transpose() + ShearCondition * d2dx0x1);
		Matrix3f df0dx2 = -k_shear * (dcdx0 * dcdx2.transpose() + ShearCondition * d2dx0x2);

		
		Matrix3f df1dx0 = -k_shear * (dcdx1 * dcdx0.transpose() + ShearCondition * d2dx1x0);
		Matrix3f df1dx1 = -k_shear * (dcdx1 * dcdx1.transpose() + ShearCondition * d2dx1x1);
		Matrix3f df1dx2 = -k_shear * (dcdx1 * dcdx2.transpose() + ShearCondition * d2dx1x2);

		Matrix3f df2dx0 = -k_shear * (dcdx2 * dcdx0.transpose() + ShearCondition * d2dx2x0);
		Matrix3f df2dx1 = -k_shear * (dcdx2 * dcdx1.transpose() + ShearCondition * d2dx2x1);
		Matrix3f df2dx2 = -k_shear * (dcdx2 * dcdx2.transpose() + ShearCondition * d2dx2x2);

		Matrix3f df0dv0 = -damping_shear * (dcdx0 * dcdx0.transpose());
		Matrix3f df0dv1 = -damping_shear * (dcdx0 * dcdx1.transpose());
		Matrix3f df0dv2 = -damping_shear * (dcdx0 * dcdx2.transpose());

		Matrix3f df1dv0 = -damping_shear * (dcdx1 * dcdx0.transpose());
		Matrix3f df1dv1 = -damping_shear * (dcdx1 * dcdx1.transpose());
		Matrix3f df1dv2 = -damping_shear * (dcdx1 * dcdx2.transpose());

		Matrix3f df2dv0 = -damping_shear * (dcdx2 * dcdx0.transpose());
		Matrix3f df2dv1 = -damping_shear * (dcdx2 * dcdx1.transpose());
		Matrix3f df2dv2 = -damping_shear * (dcdx2 * dcdx2.transpose());


		// 先搞自己的
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0,idx0) * 3) += df0dx0;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx1) * 3) += df0dx1;
		dfdx.block<3, 3>(idx0 * 3, Relative(idx0, idx2) * 3) += df0dx2;

		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx0) * 3) += df1dx0;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx1) * 3) += df1dx1;
		dfdx.block<3, 3>(idx1 * 3, Relative(idx1, idx2) * 3) += df1dx2;

		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx0) * 3) += df2dx0;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx1) * 3) += df2dx1;
		dfdx.block<3, 3>(idx2 * 3, Relative(idx2, idx2) * 3) += df2dx2;
		*/

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