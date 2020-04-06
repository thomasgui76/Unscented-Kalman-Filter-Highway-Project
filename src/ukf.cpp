#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  n_x_ = 5;
  n_aug_ = 7;
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;         // original value 30, change to 5

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/3;     // original value 30, change to M_PI/3;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  
  if(!this->is_initialized_){
    this->time_us_ = meas_package.timestamp_;
    if(meas_package.sensor_type_ == meas_package.LASER){
      double px = meas_package.raw_measurements_(0);
      double py = meas_package.raw_measurements_(1);
      this->x_ << px,py,0,0,0;
    }
    else if (meas_package.sensor_type_ == meas_package.RADAR)
    {
      double rho = meas_package.raw_measurements_(0);
      double phi = meas_package.raw_measurements_(1);
      double rho_dot = meas_package.raw_measurements_(2);
      double px = rho*cos(phi);
      double py = rho*sin(phi);
      this->x_ << px,py,rho_dot,0,0;
    }
    // initialize the covariance matrix
    MatrixXd P = MatrixXd(5,5);
    P <<   0.0,    0.0,    0.0,   0.0,   0.0,
           0.0,    0.0,    0.0,   0.0,   0.0,
           0.0,    0.0,    0.0,   0.0,   0.0,
           0.0,    0.0,    0.0,   0.0,   0.0,
           0.0,    0.0,    0.0,   0.0,   0.0;
    
    this->P_ = P;
    this->is_initialized_ = true;
  }
  else
  {
    if(meas_package.sensor_type_ == meas_package.LASER){
      if(this->use_laser_){
        std::cout<<"call lidar ProcessMeasurement: "<<std::endl;
        double delta_t = (meas_package.timestamp_ - this->time_us_)/1000000.0;     // delta time in s;
        std::cout<<"delta_t: "<<delta_t<<std::endl;
      
        this->Prediction(delta_t);
        this->UpdateLidar(meas_package);
        this->time_us_ = meas_package.timestamp_;
      }
    }
    else if (meas_package.sensor_type_ == meas_package.RADAR){
      if(this->use_radar_){
        
        std::cout<<"call radar ProcessMeasurement: "<<std::endl;
        double delta_t = (meas_package.timestamp_ - this->time_us_)/1000000.0;     // delta time in s;
        std::cout<<"delta_t: "<<delta_t<<std::endl;

        this->Prediction(delta_t);
        this->UpdateRadar(meas_package);
        this->time_us_ = meas_package.timestamp_;
      }
    }
  }
  // this->time_us_ = meas_package.timestamp_;
  
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // create augmented sigma points matrix
  MatrixXd Xsig_aug = MatrixXd(this->n_aug_,2*this->n_aug_ +1);
  this->AugmentedSigmaPoints(&Xsig_aug);
  this->SigmaPointPrediction(&Xsig_aug, delta_t);
  this->PredictMeanAndCovariance(&this->x_, &this->P_);
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // set state dimension
  int n_x = 5;
  // set augmented dimension
  int n_aug = 7;
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 2;
  // define spreading parameter
  double lambda = 3 - n_aug;
  // create radar meaurement vector z
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1);
  // set vector for weights
  VectorXd weights = VectorXd(2*n_aug+1);
  double weight_0 = lambda/(lambda+n_aug);
  double weight = 0.5/(lambda+n_aug);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug+1; ++i) {  
    weights(i) = weight;
  }
  double laspx = this->std_laspx_;
  double laspy = this->std_laspy_;
  MatrixXd Xsig_pred = this->Xsig_pred_;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  // transform sigma points into measurement space
  for(int i=0; i<2*n_aug+1; i++){
    Zsig(0,i) = Xsig_pred(0,i);
    Zsig(1,i) = Xsig_pred(1,i);
  }
  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for(int i=0; i<2*n_aug+1; i++){
      z_pred += weights(i)*Zsig.col(i);
  }
  // calculate innovation covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R << laspx*laspx, 0,
       0, laspy*laspy;

  S.fill(0.0);
  for(int i=0; i<2*n_aug+1; i++){
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S += weights(i)*z_diff*z_diff.transpose();
  }
  S += R;

  VectorXd x = this->x_;
  MatrixXd P = this->P_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i=0; i<2*n_aug+1; i++){
    VectorXd x_diff = Xsig_pred.col(i) - x;       // for Lidar, there is no angle normalization
    
    VectorXd z_diff = Zsig.col(i) - z_pred;       // for Lidar, there is no angle normalization
    
    Tc += weights(i)*x_diff*z_diff.transpose();
  }  

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z_diff = z-z_pred;
  x += K*z_diff;
  P -= K*S*K.transpose();

  // update state vector x, convariance matrix P
  this->x_ = x;
  this->P_ = P;
  
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // set state dimension
  int n_x = 5;
  // set augmented dimension
  int n_aug = 7;
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  // define spreading parameter
  double lambda = 3 - n_aug;
  // create radar meaurement vector z
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0),meas_package.raw_measurements_(1),meas_package.raw_measurements_(2);

  // set vector for weights
  VectorXd weights = VectorXd(2*n_aug+1);
  double weight_0 = lambda/(lambda+n_aug);
  double weight = 0.5/(lambda+n_aug);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug+1; ++i) {  
    weights(i) = weight;
  }

  // radar measurement noise standard deviation radius in m
  double std_radr = this->std_radr_;
  // radar measurement noise standard deviation angle in rad
  double std_radphi = this->std_radphi_;
  // radar measurement noise standard deviation radius change in m/s
  double std_radrd = this->std_radrd_;
  MatrixXd Xsig_pred = this->Xsig_pred_;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug + 1);
  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  for(int i=0; i<2*n_aug+1; i++){
      double px = Xsig_pred.col(i)(0);
      double py = Xsig_pred.col(i)(1);
      double v = Xsig_pred.col(i)(2);
      double yaw = Xsig_pred.col(i)(3);
      double yawd = Xsig_pred.col(i)(4);
      
      double rho = sqrt(px*px+py*py);
      double phi = atan2(py,px);
      double rdot = (px*cos(yaw)*v + py*sin(yaw)*v)/rho;
      
      Zsig.col(i)(0) = rho;
      Zsig.col(i)(1) = phi;
      Zsig.col(i)(2) = rdot;
  }
  // calculate mean predicted measurement
  z_pred.fill(0.0);
  for(int i=0; i<2*n_aug+1; i++){
      z_pred += weights(i)*Zsig.col(i);
  }
  
  // calculate innovation covariance matrix S
  MatrixXd R = MatrixXd(n_z,n_z);
  R << std_radr*std_radr, 0, 0,
                0, std_radphi*std_radphi, 0,
                0, 0, std_radrd*std_radrd;
                
  S.fill(0.0);
  for(int i=0; i<2*n_aug+1; i++){
      VectorXd z_diff = Zsig.col(i) - z_pred;
      if(z_diff(1)>M_PI) z_diff(1) -= 2*M_PI;
      if(z_diff(1)<-M_PI) z_diff(1) += 2*M_PI;
      S += weights(i)*z_diff*z_diff.transpose();
  }
  S += R;

  VectorXd x = this->x_;       // get the mean state vector
  MatrixXd P = this->P_;       // get the covariance matrix

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);
  // calculate cross correlation matrix
  Tc.fill(0.0);
  for(int i=0; i<2*n_aug+1; i++){
      VectorXd x_diff = Xsig_pred.col(i) - x;
      // angle normalization
      if (x_diff(3)>M_PI) x_diff(3) -= 2*M_PI;
      if (x_diff(3)<-M_PI) x_diff(3) += 2*M_PI;
      VectorXd z_diff = Zsig.col(i) -z_pred;
      // angle normalization
      if (z_diff(1)>M_PI) z_diff(1) -= 2*M_PI;
      if (z_diff(1)<-M_PI) z_diff(1) += 2*M_PI;
      
      Tc += weights(i)*x_diff*z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd z_diff = z-z_pred;
  if (z_diff(1)>M_PI) z_diff(1) -= 2*M_PI;
  if (z_diff(1)<-M_PI) z_diff(1) += 2*M_PI;
  
  x += K*z_diff;
  P -= K*S*K.transpose();
  // update state vector x, convariance matrix P
  this->x_ = x;
  this->P_ = P;
  
}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out){
  // set state dimension
  int n_x = 5;

  // set augmented dimension
  int n_aug = 7;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a = this->std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd = this->std_yawdd_;

  // define spreading parameter
  double lambda = 3 - n_aug;

  // set example state
  VectorXd x = this->x_;              // get the initial state vector x
  // get covariance matrix
  MatrixXd P = this->P_;              // get the initial covariance P
  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, 2 * n_aug + 1);
  // create augmented mean state
  x_aug.head(5) = x;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0);
  P_aug.topLeftCorner(5,5) = P;
  P_aug(5,5) = std_a * std_a;
  P_aug(6,6) = std_yawdd * std_yawdd;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for(int i=0; i<n_aug; i++){
      Xsig_aug.col(i+1) = x_aug + sqrt(lambda + n_aug) * L.col(i);
      Xsig_aug.col(i+1+n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
  }
  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_in, double delta_t){
  // set state dimension
  int n_x = 5;
  // set augmented dimension
  int n_aug = 7;
  MatrixXd Xsig_aug = *Xsig_in;
  // create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x, 2 * n_aug + 1);

  // write predicted sigma points into right column
  for(int i=0; i<2*n_aug+1; i++){
      double px = Xsig_aug(0,i);
      double py = Xsig_aug(1,i);
      double v =  Xsig_aug(2,i);
      double yaw = Xsig_aug(3,i);
      double yawd = Xsig_aug(4,i);
      double nu_a = Xsig_aug(5,i);
      double nu_yawdd = Xsig_aug(6,i);
      
      double px_p, py_p;
      if(fabs(yawd) > 0.0001){
          px_p = px + v/yawd * (sin(yaw + yawd*delta_t)-sin(yaw));
          py_p = py + v/yawd * (-cos(yaw + yawd*delta_t)+cos(yaw));
      }
      else{
          px_p = px + v*cos(yaw)*delta_t;
          py_p = py + v*sin(yaw)*delta_t;
      }
      double v_p = v;
      double yaw_p = yaw+ yawd*delta_t;
      double yawd_p = yawd;
      
      px_p += 0.5 * delta_t * delta_t*cos(yaw)*nu_a;
      py_p += 0.5 * delta_t * delta_t*sin(yaw)*nu_a;
      v_p += delta_t * nu_a;
      yaw_p += 0.5 * delta_t * delta_t * nu_yawdd;
      yawd_p += delta_t * nu_yawdd;
      
      Xsig_pred(0,i) = px_p;
      Xsig_pred(1,i) = py_p;
      Xsig_pred(2,i) = v_p;
      Xsig_pred(3,i) = yaw_p;
      Xsig_pred(4,i) = yawd_p;
  }
  this->Xsig_pred_ = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out){
  // set state dimension
  int n_x = 5;

  // set augmented dimension
  int n_aug = 7;

  // define spreading parameter
  double lambda = 3 - n_aug;
  MatrixXd Xsig_pred = this->Xsig_pred_;
  // create vector for weights
  VectorXd weights = VectorXd(2*n_aug+1);
  
  // create vector for predicted state
  VectorXd x = VectorXd(n_x);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x, n_x);
  // set weights
  double weight_0 = lambda/(lambda+n_aug);
  weights(0) = weight_0;
  for (int i=1; i<2*n_aug+1; ++i) {  // 2n+1 weights
    double weight = 0.5/(n_aug+lambda);
    weights(i) = weight;
  }
  // predict state mean
  x.fill(0.0);
  for(int i=0; i<2*n_aug+1; ++i){
      x += weights(i)*Xsig_pred.col(i);
  }

  // predict state covariance matrix
  P.fill(0.0);
  for(int i=0; i<2*n_aug+1; ++i){
      VectorXd x_diff = Xsig_pred.col(i)-x;
      if(x_diff(3)>M_PI)
        x_diff(3) -= 2*M_PI;
      if(x_diff(3)<-M_PI)
        x_diff(3) += 2*M_PI;
    
      P += weights(i)*x_diff*x_diff.transpose();
  }
  *x_out = x;
  *P_out = P;
}