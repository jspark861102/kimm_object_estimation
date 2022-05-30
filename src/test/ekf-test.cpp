#include <iostream>
#include <vector>
#include <Eigen/Dense>

// #include "kalman/main/kalman.hpp"
#include "kalman/main/extendedkalman.hpp"
#include "kalman/objdyn/object_dynamics.hpp"

int main(int argc, char* argv[]) {

  Objdyn objdyn;

  // List of noisy position measurements (y)
  std::vector<double> measurements = {
      1.04202710058, 1.10726790452, 1.2913511148, 1.48485250951, 1.72825901034,
      1.74216489744, 2.11672039768, 2.14529225112, 2.16029641405, 2.21269371128,
      2.57709350237, 2.6682215744, 2.51641839428, 2.76034056782, 2.88131780617,
      2.88373786518, 2.9448468727, 2.82866600131, 3.0006601946, 3.12920591669,
      2.858361783, 2.83808170354, 2.68975330958, 2.66533185589, 2.81613499531,
      2.81003612051, 2.88321849354, 2.69789264832, 2.4342229249, 2.23464791825,
      2.30278776224, 2.02069770395, 1.94393985809, 1.82498398739, 1.52526230354,
      1.86967808173, 1.18073207847, 1.10729605087, 0.916168349913, 0.678547664519,
      0.562381751596, 0.355468474885, -0.155607486619, -0.287198661013, -0.602973173813
  };

  /////////////////////////////////////////
  ///////////knwon parameter //////////////
  /////////////////////////////////////////  
  int n = 10; // Number of states --> object parameters
  int m = 6; // Number of measurements --> FT

  double dt = 1.0/100; // Time step

  Eigen::MatrixXd A(n, n); // System dynamics matrix
  Eigen::MatrixXd H(m, n); // Output matrix
  Eigen::MatrixXd Q(n, n); // Process noise covariance
  Eigen::MatrixXd R(m, m); // Measurement noise covariance
  Eigen::MatrixXd P(n, n); // Estimate error covariance
  Eigen::VectorXd h(m, 1); // observation
  
  A.setIdentity();         //knwon, identity
  Q.setIdentity();         //design parameter
  R.setIdentity();         //design parameter    
  P.setIdentity();         //updated parameter
  h.setZero();             //computed parameter
  H.setZero();             //computed parameter

  Q *= 0.01;
  R *= 0.05;    

  std::cout << "A: \n" << A << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "H: \n" << H << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "Q: \n" << Q << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "R: \n" << R << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "P: \n" << P << std::endl;
  std::cout << "\n" << std::endl;
  std::cout << "h: \n" << h << std::endl;
  std::cout << "\n" << std::endl;

  /////////////////////////////////////////
  ///////////initial condition ////////////
  /////////////////////////////////////////    
  Eigen::VectorXd FT_measured(m);
  Eigen::VectorXd g(3);
  Eigen::VectorXd vel(6);
  Eigen::VectorXd acc(6);
  // Eigen::MatrixXd HJacobian;  
  Eigen::VectorXd param0(n);
  double t = 0;

  g << 0, 0, 9.81;                                //if we use {tool frame}, it shoud be a variable
  vel.setZero();
  acc.setZero();
  FT_measured.setZero();  
  param0.setZero();                               //design parameter

  // Initialize the filter system
  h = objdyn.h(param0, vel, acc, g);
  H = objdyn.H(param0, vel, acc, g); 

  // Construct the filter
  EKF ekf(dt,A, H, Q, R, P, h);
  
  // Initialize the filter  
  ekf.init(t, param0);  


  /////////////////////////////////////////
  ////////////////// Run //////////////////
  /////////////////////////////////////////   
  std::cout << "t = " << t << ", " << "param_hat[0]: " << ekf.state().transpose() << std::endl;
  for(int i = 0; i < measurements.size(); i++) {
    t += dt;

    //measred variables
    FT_measured << measurements[i], measurements[i]+0.5, measurements[i]*0.2, measurements[i]*-1, measurements[i]-0.02, measurements[i]*0.02;
    vel <<        sin(2*M_PI*t),      sin(M_PI*t),      -2*sin(2*M_PI*t),       cos(M_PI*t),       0.1*cos(3*M_PI*t),         sin(5*M_PI*t);
    acc << 2*M_PI*sin(2*M_PI*t), M_PI*sin(M_PI*t), -4*M_PI*sin(2*M_PI*t),  M_PI*cos(M_PI*t),  0.3*M_PI*cos(3*M_PI*t),  5*M_PI*sin(5*M_PI*t);

    //filter system computation
    h = objdyn.h(ekf.state(), vel, acc, g);
    H = objdyn.H(ekf.state(), vel, acc, g); //ekf.state() = current estimated param    

    // ekf.update(FT_measured);
    ekf.update(FT_measured, dt, A, H, h);
    std::cout << "t = " << t << ", " << "FT_measured[" << i << "] = " << FT_measured.transpose()
        << ", param_hat[" << i << "] = " << ekf.state().transpose() << std::endl;
  }   
  return 0;
}
