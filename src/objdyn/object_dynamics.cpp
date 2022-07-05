#include <iostream>
#include <cmath>
#include <complex>
#include "kimm_object_estimation/objdyn/object_dynamics.hpp"

using namespace std;
using namespace Eigen;
using std::vector;

typedef Matrix<double, 3, 1> Vector3d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 3, 3> Matrix3d;

Objdyn::Objdyn() {}

Objdyn::~Objdyn() {}

VectorXd Objdyn::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  //Validate the estimations vector
  if(estimations.size() == 0 || estimations.size() != ground_truth.size()){
    cout<<"Error in size of Estimations vector or size mismatch with Ground Truth vector";
    return rmse;
  }
  
  //Accumulate the residual
  for(int i = 0; i < estimations.size(); ++i){
  VectorXd residual = estimations[i] - ground_truth[i];
  rmse = rmse + (residual.array() * residual.array()).matrix();
  }

  //Mean and Sqrt the error
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Objdyn::rot(const string direc, const double ang) {
  double c = cos(ang);
  double s = sin(ang);

  MatrixXd R(3,3);
  R.setIdentity();

  if(direc == "x")      R << 1,  0, 0,   0, c, -s,   0, s, c; 
  else if(direc == "y") R << c,  0, s,   0, 1,  0,  -s, 0, c; 
  else if(direc == "z") R << c, -s, 0,   s, c,  0,   0, 0, 1; 
  else cout << "Direction not recognized." << endl;

  return R;  
}

MatrixXd Objdyn::CM(const VectorXd& vec) {
  double v1 = vec(0);
  double v2 = vec(1);
  double v3 = vec(2);

  VectorXd M;
  M <<     0, -1*v3,    v2,
          v3,     0, -1*v1,
       -1*v2,    v1,     0;         

  return M;  
}

MatrixXd Objdyn::Ih(const VectorXd& vec) {  
  MatrixXd R(3,3);
  R << vec(0), vec(1), vec(2),
       vec(1), vec(3), vec(4),
       vec(2), vec(4), vec(5);

  return R;  
}

VectorXd Objdyn::h(const VectorXd& param, const Vector6d& vel, const Vector6d& acc, const Vector3d& g) {  
  // h : observation model
  
  // param = [m, cx, cy, cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
  // r_state = [drx, dry, drz, ddrx, ddry, ddrz, ddx, ddy, ddz, rx, ry, rz]
  // f & t is force at the tool coordinate. If we have force at the robot base coordinate, we should transform force from {base} to {tool}  
  // ---> no, {base} is bette

  // r^fo = m(ar + alpha*r^Po + omega*(omgega*r^Po) - g)
  // r^to = Ir'alpha + omega*(Ir'omega) + m'r^Po*(ar - g)
  // whre r^fo = fr + fh
  //      r^to = tr + th  - h^Pr*fh
  //      Ir = Io + m'r^Po*r^Po
  // threfore,
  //      measred : ar, alpha, omega, (vr), fr, tr
  //      unkown (or measured) : fh, th, h^Pr
  //      to be identified : x

  // h = [m(ar + alpha*r^Po + omega*(omgega*r^Po) - g);
  //      Ir'alpha + omega*(Ir'omega) + m'r^Po*(ar - g)]  

  double mass;
  Vector3d com(3);
  Matrix3d Ir(3,3);
  Vector3d ar(3);
  Vector3d alpha(3);
  Vector3d vr(3);
  Vector3d omega(3);

  VectorXd FT_estimated(6);

  mass = param(0);
  com << param(1), param(2), param(3);
  Ir = Ih(param.tail(6));
  ar = acc.head(3);
  alpha = acc.tail(3);
  vr = vel.head(3);
  omega = vel.tail(3);

  FT_estimated.head(3) = mass*(ar + alpha.cross(com) + omega.cross(omega.cross(com)) - g);
  FT_estimated.tail(3) = Ir*alpha + omega.cross(Ir*omega) + mass*com.cross(ar-g);

  return FT_estimated; 
}

MatrixXd Objdyn::H(const VectorXd& param, const Vector6d& vel, const Vector6d& acc, const Vector3d& g) {  
  // H : jacobian of h (partial derivative of h w.r.t state)
  // Here, complex step differentiation is used
  //  F(x0 + ih) = F(x0) + ihF'(x0) - (h^2 * F''(x0))/(2!) -  (ih^3 * F'''(x0))/(3!) + ...
  //  F'(x0) =Im(F(x-+ih))/h + h.o.t
  
  double n = param.size(); //10
  double m = h(param, vel, acc, g).size(); //6

  MatrixXd HJacobian;    
  HJacobian.resize(m,n);
  double step = 0.00000001; //10^-8
  
  VectorXd param_h;  
  param_h.resize(n);
  for(int k=0; k<n; ++k) {
    param_h = param;
    param_h(k) = param(k) + step;

    HJacobian.block(0,k,m,1) = ( h(param_h, vel, acc, g) - h(param, vel, acc, g) ) / step; //HJacobian(:,k)

    // cout << "here" << endl;
    // cout << HJacobian.block(0,k,m,1) << endl;
    // cout << HJacobian.block(0,k,m,0) << endl;
    // cout << HJacobian.rows() << "  " << HJacobian.cols() << endl;
  }

  // complex<double> step;
  // step = 0.0d + 0.00000001id;  

  // MatrixXcd A;
  // A.resize(m,n);
  // // complex<double> Cparam;
  // VectorXcd Cparam;  
  // for(int k=0; k < n; ++k) {
  //   Cparam = param;
  //   Cparam(k) = param(k)*1.0d + step;
    
  //   // h(Cparam, FT_estimated, FT_estimated, FT_estimated.head(3));

  //   // A.block(0,k,m,1) = Cparam;      


  // }    
  return HJacobian; 
}
