#ifndef OBJECT_DYNAMICS_H_
#define OBJECT_DYNAMICS_H_
#include <vector>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

typedef Matrix<double, 3, 1> Vector3d;
typedef Matrix<double, 6, 1> Vector6d;


class Objdyn {
public:
  /**
  * Constructor.
  */
  Objdyn();

  /**
  * Destructor.
  */
  virtual ~Objdyn();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * object dynamics equations
  */
  MatrixXd rot(const string direc, const double ang);
  MatrixXd CM(const VectorXd& vec);
  MatrixXd Ih(const VectorXd& vec);  
  VectorXd h(const VectorXd& param, const Vector6d& vel, const Vector6d& acc, const Vector3d& g); 
  MatrixXd H(const VectorXd& param, const Vector6d& vel, const Vector6d& acc, const Vector3d& g);

};

#endif /* OBJECT_DYNAMICS_ */
