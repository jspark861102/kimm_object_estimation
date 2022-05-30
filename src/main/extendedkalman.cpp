#include <iostream>
#include <stdexcept>

#include "kalman/main/extendedkalman.hpp"

EKF::EKF(
    double dt,
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& P,
    const Eigen::VectorXd& h)
  : A(A), H(H), Q(Q), R(R), P0(P), h(h),
    m(H.rows()), n(A.rows()), dt(dt), initialized(false),
    I(n, n), x_hat(n), x_hat_new(n)
{
  I.setIdentity();
}

EKF::EKF() {}

void EKF::init(double t0, const Eigen::VectorXd& x0) {
  x_hat = x0;
  P = P0;
  this->t0 = t0;
  t = t0;
  initialized = true;
}

void EKF::init() {
  x_hat.setZero();
  P = P0;
  t0 = 0;
  t = t0;
  initialized = true;
}

void EKF::update(const Eigen::VectorXd& y) {

  if(!initialized)
    throw std::runtime_error("Filter is not initialized!");
  //prediction
  x_hat_new = A * x_hat;
  P = A*P*A.transpose() + Q;

  //correction
  K = P*H.transpose()*(H*P*H.transpose() + R).inverse();
  x_hat_new += K * (y - h);
  P = (I - K*H)*P;
  x_hat = x_hat_new;

  t += dt;
}

void EKF::update(const Eigen::VectorXd& y, double dt, const Eigen::MatrixXd A, const Eigen::MatrixXd H, const Eigen::VectorXd h) {

  this->A = A;
  this->H = H;
  this->h = h;
  this->dt = dt;
  update(y);
}
