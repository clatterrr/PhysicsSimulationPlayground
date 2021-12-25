// ch05 使用freeglut 来操作opengl -- 动态三角形
#include <iostream>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include <Eigen\Core>
#include "Condition.h"
#include "cgsolver.h"

using namespace Eigen;

float dx = 1 / (Nx - 1);
float dt = 5.0f;
float mass = 10;
Vector3f node_pos[node_num];
Vector3f node_vel[node_num];
bool node_fixed[node_num];
const int element_num = (Nx - 1) * (Ny - 1) * 2;
int element[element_num * 3];

Matrix2f element_duv[element_num];
Vector3f element_dwu[element_num];
Vector3f element_dwv[element_num];
Vector3f light_color(1.0, 1.0, 1.0);
Vector3f light_pos(1, 1, 1);

Condition cond;
cgsolver solver(node_num,Nx);
float static depth = 1.0f;



Vector3f CrossMe(Vector3f p0, Vector3f p1)
{
	Vector3f result;
	result(0) = p0(1) * p1(2) - p0(2) * p1(1);
	result(1) = p0(2) * p1(0) - p0(0) * p1(2);
	result(2) = p0(0) * p1(1) - p0(1) * p1(0);
	return result;
}

void TriangleWithLight(Vector3f p0, Vector3f p1, Vector3f p2)
{

	// 简单的逐顶点光照
	Vector3f e10 = p1 - p0;
	Vector3f e20 = p2 - p0;
	Vector3f normal = CrossMe(e10, e20).normalized();
	Vector3f dir = (p0 - light_pos).normalized();
	float intensity = dir.dot(normal);
	Vector3f color = intensity * light_color;
	glColor3f(color(0), color(1), color(2));
	glVertex3f(p0(0), p0(1), p0(2));

	dir = (p1 - light_pos).normalized();
	intensity = dir.dot(normal);
	color = intensity * light_color;
	glColor3f(color(0), color(1), color(2));
	glVertex3f(p1(0), p1(1), p1(2));

	dir = (p2 - light_pos).normalized();
	intensity = dir.dot(normal);
	color = intensity * light_color;
	glColor3f(color(0), color(1), color(2));
	glVertex3f(p2(0), p2(1), p2(2));
}

void DrawGround()
{
	glBegin(GL_TRIANGLES);

	Vector3f p1(-1, -1, 0);
	Vector3f p2(-1, -1, -2);
	Vector3f p3(1, -1, -2);
	Vector3f p4(1, -1, 0);

	TriangleWithLight(p1, p2, p3);
	TriangleWithLight(p1, p3, p4);
	glEnd();
}

void DrawCloth()
{
	glBegin(GL_TRIANGLES);
	for (int it = 0; it < element_num; it++)
	{
		TriangleWithLight(node_pos[element[it * 3 + 0]], node_pos[element[it * 3 + 1]], node_pos[element[it * 3 + 2]]);
	}

	glEnd();
}

void RenderMainProgram() {

	// 清空颜色缓存
	glClear(GL_COLOR_BUFFER_BIT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluLookAt(-1,0,0,0,0,0,0,0,1);
	//glFrustum(-1, 1, -1, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();

	gluPerspective(60.0f, 1.0f, 1.0f, 100.0f);
	glTranslatef(0.0f, 0.0f, -2.0f);
	glRotated(10.0, 1, 0, 0);


	DrawGround();

	DrawCloth();

	glPopMatrix();

	glutSwapBuffers();
}

VectorXf preSolve()
{
	// ref . Large Step In Cloth Simulation eqn.6
#ifdef SPARSE_DF
	lhs = MatrixXf::Zero(node_num * 3, 9 * 3);
#else
	lhs = MatrixXf::Zero(node_num * 3, node_num * 3);
#endif
	VectorXf v;
	v = VectorXf::Zero(node_num * 3);
	for (int i = 0; i < node_num; i++)
	{
#ifdef SPARSE_DF
		lhs.block<3, 3>(i * 3, 0) = Matrix3f::Identity(3, 3);
#else
		lhs.block<3, 3>(i * 3, i * 3) = Matrix3f::Identity(3, 3);
#endif
		v.segment<3>(i * 3) = node_vel[i];
	}
	lhs = lhs - dx / mass * dfdv - dx * dx / mass * dfdx;
	rhs = VectorXf::Zero(node_num * 3);
#ifdef SPARSE_DF
	for (int i = 0; i < node_num; i++)
	{
		rhs.segment<3>(i * 3) += dx / mass * (forces.segment<3>(i * 3));
		for (int j = 0; j < 9; j++)
		{
			int idx = solver.colToIdx(i, j);
			if (idx != -1)
			{
				rhs.segment<3>(i * 3) += dx / mass * dx * dfdx.block<3, 3>(i*3, j*3) * v.segment<3>(idx*3);
			}
		}
	}
#else
	rhs = dx / mass * (forces + dx * dfdx * v);
#endif

	VectorXf result = VectorXf::Zero(node_num * 3);
#ifdef SPARSE_DF
	result = solver.solve();
#else
	result = lhs.inverse() * rhs;
#endif

	/*
	std::cout << "dfdx" << std::endl;
	std::cout << dfdx << std::endl;
	std::cout << "dfdv" << std::endl;
	std::cout << dfdv << std::endl;
	std::cout << "result" << std::endl;
	std::cout << result << std::endl;
	std::cout << "forces" << std::endl;
	std::cout << forces << std::endl;
	*/
	return result;
}

void subStep()
{
	Vector3f wu;
	Vector3f wv;
	Matrix2f duv_inv;
	float det;
	Vector3f p0, p1, p2, p3;
	Vector3f dp10;
	Vector3f dp20;
	int idx0, idx1, idx2, idx3;
#ifdef SPARSE_DF
	dfdx = MatrixXf::Zero(node_num * 3, 9 * 3);
	dfdv = MatrixXf::Zero(node_num * 3, 9 * 3);
#else
	dfdx = MatrixXf::Zero(node_num * 3, node_num * 3);
	dfdv = MatrixXf::Zero(node_num * 3, node_num * 3);
#endif
	for (int i = 0; i < node_num; i++)
	{
		forces[i * 3 + 0] = 0;
		forces[i * 3 + 1] = -0.05;
		forces[i * 3 + 2] = 0;
	}
	for (int i = 0; i < element_num; i++)
	{

		duv_inv = element_duv[i];

		idx0 = element[i * 3 + 0];
		idx1 = element[i * 3 + 1];
		idx2 = element[i * 3 + 2];

		p0 = node_pos[idx0];
		p1 = node_pos[idx1];
		p2 = node_pos[idx2];
		dp10 = p1 - p0;
		dp20 = p2 - p0;


		// 矩阵默认按列存储
		wu = dp10 * duv_inv(0, 0) + dp20 * duv_inv(1, 0);
		wv = dp10 * duv_inv(0, 1) + dp20 * duv_inv(1, 1);
		cond.ComputeShearForces(wu, wv, element_dwu[i], element_dwv[i],
			p0, p1, p2, node_vel[idx0], node_vel[idx1], node_vel[idx2], idx0, idx1, idx2);
		cond.ComputeStretchForces(wu, wv, element_dwu[i], element_dwv[i],
			p0, p1, p2, node_vel[idx0], node_vel[idx1], node_vel[idx2], idx0, idx1, idx2);
		if (i % 2 == 0)
		{
			idx0 = element[i * 3 + 0];
			idx1 = element[i * 3 + 2];
			idx2 = element[i * 3 + 1];
			idx3 = element[i * 3 + 4];
			p0 = node_pos[idx0];
			p1 = node_pos[idx1];
			p2 = node_pos[idx2];
			p3 = node_pos[idx3];
			cond.computeBendingForce(p0, p1, p2, p3, 
				node_vel[idx0], node_vel[idx1], node_vel[idx2], node_vel[idx3], idx0, idx1, idx2, idx3);
		}
		//break;

	}
	VectorXf dv = preSolve();
	for (int i = 0; i < node_num; i++)
	{
		if (node_fixed[i])continue;
		//node_vel[i] += dt * forces.segment<3>(i * 3) / mass;
		node_vel[i] += dv.segment<3>(i * 3);
		node_pos[i] += dt * node_vel[i];
	}



}


void timer(int junk)
{
	subStep();
	glutPostRedisplay();
	glutTimerFunc(30, timer, 0);
}

void InitArray()
{
	int cnt = 0;
	for (int j = 0; j < Ny; j++)
	{
		for (int i = 0; i < Nx; i++)
		{
			node_pos[cnt] = Vector3f(i * dx - 0.5, j * dx - 0.5, -1);
			node_vel[cnt] = Vector3f(0, 0, 0);
			node_fixed[cnt] = false;
			if (j == Nx - 1)
				node_fixed[cnt] = true;
			cnt += 1;
		}
	}
	cnt = 0;
	node_pos[0](2) += 0.5;
	Vector2f duv10, duv20;
	for (int j = 0; j < Ny - 1; j++)
	{
		for (int i = 0; i < Nx - 1; i++)
		{
			int idx = j * Nx + i;
			element[cnt * 6 + 0] = idx;
			element[cnt * 6 + 1] = idx + Nx;
			element[cnt * 6 + 2] = idx + 1;

			element[cnt * 6 + 3] = idx + Nx;
			element[cnt * 6 + 4] = idx + Nx + 1;
			element[cnt * 6 + 5] = idx + 1;

			// 目前所有三角形的uv都是一样的，但我还是把计算公式在这里
			//2  --- 3
			// |   \   |
			// 0 --- 1
			duv10 = Vector2f(0, dx);
			duv20 = Vector2f(dx, 0);
			float det = 1 / (duv10(0) * duv20(1) - duv10(1) * duv20(0));
			element_duv[cnt * 2] << duv20(1), -duv20(0),
				-duv10(1), duv10(0);
			element_duv[cnt * 2] *= det;

			element_dwu[cnt * 2](0) = (duv10(1) - duv20(1)) * det;
			element_dwu[cnt * 2](1) = duv20(1) * det;
			element_dwu[cnt * 2](2) = -duv10(1) * det;

			element_dwv[cnt * 2](0) = (duv20(0) - duv10(0)) * det;
			element_dwv[cnt * 2](1) = -duv20(0) * det;
			element_dwv[cnt * 2](2) = duv10(0) * det;

			duv10 = Vector2f(dx, 0);
			duv20 = Vector2f(dx, -dx);
			det = 1 / (duv10(0) * duv20(1) - duv10(1) * duv20(0));
			element_duv[cnt * 2 + 1] << duv20(1), -duv20(0),
				-duv10(1), duv10(0);
			element_duv[cnt * 2 + 1] *= det;

			element_dwu[cnt * 2 + 1](0) = (duv10(1) - duv20(1)) * det;
			element_dwu[cnt * 2 + 1](1) = duv20(1) * det;
			element_dwu[cnt * 2 + 1](2) = -duv10(1) * det;

			element_dwv[cnt * 2 + 1](0) = (duv20(0) - duv10(0)) * det;
			element_dwv[cnt * 2 + 1](1) = -duv20(0) * det;
			element_dwv[cnt * 2 + 1](2) = duv10(0) * det;

			cnt += 1;
		}
	}
}

int main(int argc, char** argv) {

	// 初始化GLUT
	glutInit(&argc, argv);
	// 显示模式：双缓冲、RGBA
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

	// 窗口设置
	glutInitWindowSize(512, 512);      // 窗口尺寸
	glutInitWindowPosition(300, 300);  // 窗口位置
	glutCreateWindow("Tutorial 01");   // 窗口标题

	glutTimerFunc(1000, timer, 0);
	glutDisplayFunc(RenderMainProgram);

	// 缓存清空后的颜色值
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	glewInit();
	InitArray();


	// glut主循环
	glutMainLoop();
	return 1;
}