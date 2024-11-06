#include <math.h>
#include <RcppArmadillo.h>
#include <RcppEnsmallen.h>
using namespace Rcpp;

// [[Rcpp::depends("RcppEnsmallen")]]
// [[Rcpp::depends("RcppArmadillo")]]


///////// general functions /////////

// maximum exponent to avoid numerical instability //
const double MAX_EXP = 15;
const double MinGradientNorm = 1e-6;
const double Factr = 1e-10;

// matrix inverse //
arma::mat myinvCpp(const arma::mat& A){
  bool invflag = false;
  arma::mat B = A;
  invflag = arma::inv_sympd(B , A, arma::inv_opts::allow_approx);
  if(!invflag){
    //Rcout << "inv_sympd failed, try inv\n";
    invflag = arma::inv( B, A, arma::inv_opts::allow_approx);
    if(!invflag){
      //Rcout << "inv failed, try pinv\n";
      invflag = arma::pinv( B,A);
      if(!invflag){
        //Rcout << "all inv methods failed!\n";
        throw std::runtime_error("error");
      }
    }
  }
  return B;
}

// Cholesky decomposition //
arma::mat myCholCpp(arma::mat A){
  bool flag = false;
  arma::mat B( arma::size(A), arma::fill::zeros);
  flag = arma::chol(B , A, "lower");
  if(!flag){
    arma::vec avec = A.diag();
    arma::vec tmp = avec.elem(arma::find(avec));
    double val = 0.01 * arma::mean(arma::abs(tmp));
    A.diag() += val;
    flag = arma::chol(B , A,"lower");
    if(!flag){
      B.diag().fill(val);
    }
  }
  return B;
}


// matrix lower-triangular matrix
arma::mat makeLowTriMat(const arma::mat& V,
                        const arma::vec& Lvec){
  arma::uvec lower_indices = arma::trimatl_ind( arma::size(V) );
  arma::mat L( arma::size(V), arma::fill::zeros);
  L(lower_indices) = Lvec;
  return L;
}

// extract lower-triangular elements of matrix
arma::vec LowTriVec(const arma::mat& V){
  arma::uvec lower_indices = arma::trimatl_ind( arma::size(V) );
  arma::vec Lvec = V(lower_indices);
  return Lvec;
}


// reshape field of vec to n \times K
void field_reshape_vec(const arma::field<arma::vec>& Y_tmp,
                       arma::field<arma::vec>& Y, int n, int K ){
  int iter = 0;
  for(int k=0; k<K;k++){
    for(int i=0; i<n; i++){
      Y(i,k) = Y_tmp(iter);
      iter++;
    }
  }
}

// reshape field of mat to n \times K
void field_reshape_mat(const arma::field<arma::mat>& Y_tmp,
                       arma::field<arma::mat>& Y, int n, int K ){
  int iter = 0;
  for(int k=0; k<K;k++){
    for(int i=0; i<n; i++){
      Y(i,k) = Y_tmp(iter);
      iter++;
    }
  }
}

// convert field of vec to vec
arma::vec field_to_alpha_vec(const arma::field<arma::vec>& X_T,
                             const arma::vec& alpha,
                             int i_now, arma::uvec p_x_vec,
                             arma::uvec idx){

  //arma::uvec idx = arma::find(alpha);
  int p_x = arma::accu(p_x_vec.elem(idx));
  arma::vec X_ia(p_x);

  int start = 0, k;
  for(int j=0; j<idx.n_elem; j++){
    k = idx(j);
    X_ia.subvec(start,start+p_x_vec(k)-1) = X_T(i_now, k) * alpha(k);
    start = start+p_x_vec(k);
  }
  return X_ia;
}

// convert field of vec to vec
arma::vec field_to_alpha_vec_full(const arma::field<arma::vec>& X_T,
                             const arma::vec& alpha,
                             int i_now, arma::uvec p_x_vec){

  //arma::uvec idx = arma::find(alpha);
  int p_x = arma::accu(p_x_vec);
  arma::vec X_ia(p_x);

  int start = 0;
  for(int k=0; k<alpha.n_elem; k++){
    X_ia.subvec(start,start+p_x_vec(k)-1) = X_T(i_now, k) * alpha(k);
    start = start+p_x_vec(k);
  }
  return X_ia;
}

// convert field of mat to mat
arma::mat field_to_alpha_mat(const arma::field<arma::mat>& X_t,
                             const arma::vec& alpha,
                             int i_now, const arma::uvec& p_x_vec,
                             arma::uvec idx){

  //arma::uvec idx = arma::find(alpha);
  int p_x = arma::accu(p_x_vec.elem(idx));
  arma::mat X_ia(X_t(i_now,0).n_rows , p_x);

  int start = 0,k;
  for(int j=0; j<idx.n_elem; j++){
    k = idx(j);
    X_ia.cols(start,start+p_x_vec(k)-1) = X_t(i_now, k) * alpha(k);
    start = start+p_x_vec(k);
  }
  return X_ia;
}

// convert field of mat to mat
arma::mat field_to_alpha_mat_full(const arma::field<arma::mat>& X_t,
                             const arma::vec& alpha,
                             int i_now, const arma::uvec& p_x_vec){

  //arma::uvec idx = arma::find(alpha);
  int p_x = arma::accu(p_x_vec);
  arma::mat X_ia(X_t(i_now,0).n_rows , p_x);

  int start = 0;
  for(int k=0; k<alpha.n_elem; k++){
    X_ia.cols(start,start+p_x_vec(k)-1) = X_t(i_now, k) * alpha(k);
    start = start+p_x_vec(k);
  }
  return X_ia;
}

arma::mat field_to_zero_mat_full(const arma::field<arma::mat>& X_t,
                                  const arma::vec& alpha,
                                  int i_now, const arma::uvec& p_x_vec,
                                  int k_now){

  //arma::uvec idx = arma::find(alpha);
  int p_x = arma::accu(p_x_vec);
  arma::mat X_ia(X_t(i_now,0).n_rows , p_x, arma::fill::zeros);

  int start = 0;
  for(int k=0; k<alpha.n_elem; k++){
    if(k == k_now){
      //X_ia.cols(start,start+p_x_vec(k)-1) = X_t(i_now, k) * alpha(k);
      X_ia.cols(start,start+p_x_vec(k)-1) = X_t(i_now, k);
      break;
    }
    start = start+p_x_vec(k);
  }
  return X_ia;
}

arma::mat field_to_zero_noalpha_mat_full(const arma::field<arma::mat>& X_t,
                                 int i_now, const arma::uvec& p_x_vec,
                                 int k_now){

  //arma::uvec idx = arma::find(alpha);
  int p_x = arma::accu(p_x_vec);
  arma::mat X_ia(X_t(i_now,0).n_rows, p_x, arma::fill::zeros);

  int start = 0;
  for(int k=0; k < p_x_vec.n_elem; k++){
    if(k == k_now){
      X_ia.cols(start,start+p_x_vec(k)-1) = X_t(i_now, k);
      break;
    }
    start = start+p_x_vec(k);
  }
  return X_ia;
}

// convert field of mat to field of diagonal matrix
arma::field<arma::mat> field_to_field_Dmat(
    const arma::field<arma::mat>& X_t, const arma::uvec& p_x_vec){

  int p_x = arma::accu(p_x_vec);
  int K = p_x_vec.n_elem;
  int nt = X_t(0).n_rows;

  arma::field<arma::mat> X_D = arma::field<arma::mat>(nt);

  for(int j = 0; j<nt; j++){

    int start = 0, end=0;
    X_D(j) = arma::mat(p_x,K,arma::fill::zeros);
    for(int k=0; k<K; k++){
      end = start+p_x_vec(k)-1;
      X_D(j)(arma::span(start,end), k)= X_t(k).row(j).t();
      start = end +1;
    }
  }

  return X_D;
}

arma::mat field_to_Dmat(const arma::field<arma::vec>& X_T,
                        const arma::uvec& p_x_vec){

  int p_x = arma::accu(p_x_vec);
  int K = p_x_vec.n_elem;

  arma::mat X_D = arma::mat(p_x,K,arma::fill::zeros);
  int start = 0, end=0;

  for(int k=0; k<K; k++){
    end = start+p_x_vec(k)-1;
    X_D(arma::span(start,end), k) = X_T(k);
    start = end +1;
  }
  return X_D;
}


// convert vec to field of vec
arma::field<arma::vec> vec_to_field(const arma::vec& mu,
                                    const arma::uvec& p_z_vec){

  arma::field<arma::vec> mu_f(p_z_vec.n_elem);

  int start = 0;
  for(int k=0; k<p_z_vec.n_elem; k++){
    mu_f(k) = mu.subvec(start,start+p_z_vec(k)-1);
    start = start+p_z_vec(k);
  }

  return mu_f;
}

// convert vec to field of vec
arma::field<arma::vec> vec_to_field_L(const arma::vec& L,
                                      const arma::uvec& p_z_vec){

  arma::field<arma::vec> L_f(p_z_vec.n_elem);

  int start = 0,step;
  for(int k=0; k<p_z_vec.n_elem; k++){
    step = p_z_vec(k)*(p_z_vec(k)+1)/2;
    L_f(k) = L.subvec(start,start+step-1);
    start = start + step ;
  }
  return L_f;
}

// convert  field of vec  to vec
arma::vec field_to_vec(const arma::field<arma::vec>& mu,
                       const arma::uvec& p_z_vec){

  int p_z = arma::accu(p_z_vec);
  arma::vec mu_vec(p_z);

  int start = 0;
  for(int k=0; k<p_z_vec.n_elem; k++){
    mu_vec.subvec(start,start+p_z_vec(k)-1) = mu(k);
    start = start+p_z_vec(k);
  }

  return mu_vec;
}

// extract lower-triangular elements of field of matrices
arma::vec LowTriVec_field(const arma::field<arma::mat>& V){
  arma::uvec p_vec(V.n_elem);
  arma::field<arma::vec> Lvec_f(V.n_elem);

  for(int j =0; j< V.n_elem; j++){
    arma::uvec lower_indices = arma::trimatl_ind( arma::size(V(j)) );
    Lvec_f(j) = V(j)(lower_indices);
    p_vec(j) = Lvec_f(j).n_elem;
  }

  return field_to_vec(Lvec_f, p_vec);
}


//////// functions for CoxFM ////////

// CoxFM data struct
struct CoxFM_data_t{
  // data part //
  arma::field<arma::mat> X; // n \times 1 mat
  arma::field<arma::mat> Z; // n \times 1 mat
  arma::field<arma::mat> X_t; // n \times 1 mat
  arma::field<arma::mat> Z_t; // n \times 1 mat
  arma::field<arma::vec> GQ_w; // n \times 1 vec, Gauss-quadrature weights
  arma::field<arma::vec> GQ_t; // n \times 1 vec, Gauss-quadrature nodes

  int n; //total number of subjects
  int p_z; // total number of random effects

  // initialization function //
  CoxFM_data_t(const List& datalist)
  {
    arma::field<arma::vec> GQ_w_tmp = datalist["GQ_w"];
    GQ_w = GQ_w_tmp;
    GQ_w_tmp.clear();
    arma::field<arma::vec> GQ_t_tmp = datalist["GQ_t"];
    GQ_t = GQ_t_tmp;
    GQ_t_tmp.clear();

    n = GQ_w.n_elem;

    arma::field<arma::mat> X_tmp = datalist["X"];
    X = X_tmp;
    X_tmp.clear();

    arma::field<arma::mat> Z_tmp = datalist["Z"];
    Z = Z_tmp;
    Z_tmp.clear();

    p_z = Z(0).n_cols;

    arma::field<arma::mat> X_t_tmp = datalist["X_t"];
    X_t = X_t_tmp;
    X_t_tmp.clear();

    arma::field<arma::mat> Z_t_tmp = datalist["Z_t"];
    Z_t = Z_t_tmp;
    Z_t_tmp.clear();
  }

};

// CoxFM parameter struct
struct CoxFM_para_t{
  // para part //
  arma::field<arma::vec> mu; // n \times 1 vec
  arma::field<arma::mat> V; // n \times 1 mat
  arma::field<arma::vec> Lvec; // n \times 1 vec: Lvec*Lvec.t() = V

  arma::vec beta; // p \times 1 vec
  arma::mat Sigma; // dim = (q \times q)
  arma::mat invSigma; // inverse of Sigma

  // initialization function //
  CoxFM_para_t(const List& paralist)
  {
    beta = as<arma::vec>(paralist["beta"]);
    Sigma = as<arma::mat>(paralist["Sigma"]);
    invSigma = myinvCpp(Sigma);

    arma::field<arma::mat> V_tmp = paralist["V"];
    V = V_tmp;
    V_tmp.clear();
    int n = V.n_elem;
    Lvec = arma::field<arma::vec>(n);
    for(int i=0; i<n; i++){
      // Cholesky decomposition
      //arma::mat Ltmp = arma::chol(V(i),"lower");
      arma::mat Ltmp = myCholCpp(V(i));
      arma::uvec lower_indices = arma::trimatl_ind(arma::size(Ltmp));
      Lvec(i) = Ltmp(lower_indices);
    }

    arma::field<arma::vec> mu_tmp = paralist["mu"];
    mu = mu_tmp;
    mu_tmp.clear();

  }

  void updateInvSigma(){
    invSigma = myinvCpp(Sigma);
  }

};


// calculate ELBO //
double CoxFM_calcELBO(const CoxFM_data_t& data,
                      const CoxFM_para_t& para){

  //double ELBO=0;
  arma::vec ELBO(data.n,arma::fill::zeros);

  for(int i=0; i< data.n; i++){

    ////
    ELBO(i) += arma::accu(data.X(i) * para.beta + data.Z(i) * para.mu(i));

    ////
    arma::vec h_it = data.X_t(i) * para.beta + data.Z_t(i) * para.mu(i);
    for(int j=0; j< h_it.n_elem; j++){
      h_it(j) += 0.5 * arma::as_scalar(
        data.Z_t(i).row(j) * para.V(i) * data.Z_t(i).row(j).t()
      );
    }

    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);
    ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

    ////
    double val, sign;
    arma::log_det(val, sign, para.Sigma);
    ELBO(i) -= 0.5 * val;
    ELBO(i) -= 0.5 * arma::as_scalar(para.mu(i).t()*para.invSigma*para.mu(i));
    ELBO(i) -= 0.5 * arma::trace(para.invSigma * para.V(i));

    arma::log_det(val, sign, para.V(i));
    ELBO(i) += 0.5 * val;
  }

  return arma::accu(ELBO);
}

// update  Sigma //
void CoxFM_updateSig(const CoxFM_data_t& data,
                     CoxFM_para_t& para){

  arma::mat Sigma_tmp(arma::size(para.Sigma), arma::fill::zeros);

  for(int i=0; i< data.n; i++){
    Sigma_tmp += para.mu(i) *  para.mu(i).t() + para.V(i);
  }

  Sigma_tmp /= data.n;
  para.Sigma = Sigma_tmp;
  para.invSigma = myinvCpp(Sigma_tmp);
}

// update beta //
class CoxFM_updateBetaFun{
public:
  const CoxFM_data_t& data;
  const CoxFM_para_t& para;

  CoxFM_updateBetaFun(const CoxFM_data_t& data,
                      const CoxFM_para_t& para):
    data(data), para(para){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& beta_t, arma::mat& g)
  {

    arma::vec beta = beta_t.col(0);
    arma::mat grad_beta(beta.n_elem, data.n, arma::fill::zeros);
    arma::vec ELBO(data.n, arma::fill::zeros);

    for(int i=0; i< data.n; i++){

      // calculate function value
      ////
      ELBO(i) += arma::accu(data.X(i) * beta);

      ////
      arma::vec h_it = data.X_t(i) * beta + data.Z_t(i) * para.mu(i);
      for(int j=0; j< h_it.n_elem; j++){
        h_it(j) += 0.5 * arma::as_scalar(
          data.Z_t(i).row(j) * para.V(i) * data.Z_t(i).row(j).t()
        );
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

      // gradient of beta //
      arma::vec grad_beta_tmp = arma::sum(data.X(i).t(),1);
      grad_beta_tmp -= data.X_t(i).t() * (data.GQ_w(i) % h_it);
      grad_beta.col(i) = grad_beta_tmp;
    }

    double fval= -1*arma::accu(ELBO)/data.n;
    g.col(0) = -1*  arma::sum(grad_beta,1)/data.n;

    return fval;
  }

};

// update variational parameters mu_i and V_i //
class CoxFM_updateMuVFun{
public:
  const CoxFM_data_t& data;
  const CoxFM_para_t& para;
  int i=0;

  CoxFM_updateMuVFun(const CoxFM_data_t& data,
                     const CoxFM_para_t& para) :
    data(data), para(para){

  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& muV, arma::mat& g)
  {
    arma::vec mu = muV(arma::span(0,data.p_z-1), 0);
    arma::vec Lvec = muV(arma::span(data.p_z, muV.n_rows-1), 0);

    arma::mat L =  makeLowTriMat(para.V(i),  Lvec);
    arma::mat V = L * L.t();

    double val;
    double sign;


    /// fun value
    double fval = 0.0;
    arma::vec h_it = data.X_t(i) * para.beta + data.Z_t(i) * mu;
    for(int j=0; j< h_it.n_elem; j++){
      h_it(j) += 0.5 * arma::as_scalar(
        data.Z_t(i).row(j) * V * data.Z_t(i).row(j).t()
      );
    }
    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);

    fval += arma::accu(data.Z(i) * mu) -
      arma::accu(data.GQ_w(i) % h_it) -
      0.5 * arma::as_scalar(mu.t() * para.invSigma *mu);

    arma::log_det(val, sign, V);
    fval +=  0.5 * val - 0.5 * arma::trace(para.invSigma * V);

    /// gradient

    arma::mat grad_V = -1*para.invSigma*L +
      arma::trans(arma::inv( arma::trimatl(L)));

    grad_V -= data.Z_t(i).t()*arma::diagmat((data.GQ_w(i) % h_it)) *
      data.Z_t(i) * L;


    arma::vec grad_mu(mu.n_rows,arma::fill::zeros);

    grad_mu +=  arma::sum(data.Z(i).t(),1) -
      data.Z_t(i).t() * (data.GQ_w(i) % h_it) -
      para.invSigma * mu;

    fval = -1*fval;

    g(arma::span(0,data.p_z-1),0) = -grad_mu;

    g(arma::span(data.p_z,muV.n_rows-1),0) = -LowTriVec(grad_V);

    //Rcout << fval << "\n";

    return fval;
  }

};

// to put the new updates into para //
void CoxFM_storeMuV(CoxFM_para_t& para, const arma::vec& mu,
                    const arma::vec& Lvec, const int& i){

  para.Lvec(i) = Lvec;
  arma::mat L =  makeLowTriMat( para.V(i),  Lvec);
  para.V(i) = L*L.t();
  para.mu(i) = mu;
}

// combine all parameters into a vector
arma::vec CoxFM_combinaPara(const CoxFM_para_t& para){
  arma::vec sig_vec = LowTriVec(para.Sigma);
  return arma::join_cols(para.beta, sig_vec);
}

//' Main function to run CoxFM
//' @noRd
//'
// [[Rcpp::export]]
List init_CoxFM(const List& datalist, const List& paralist,
                int maxiter = 100, double eps=1e-4){

  //Rcout << "1\n";
  CoxFM_data_t data(datalist);

  //Rcout << "2\n";
  CoxFM_para_t para(paralist);

  ens::L_BFGS lbfgs;
  lbfgs.MinGradientNorm() = MinGradientNorm;
  lbfgs.Factr() = Factr;

  CoxFM_updateMuVFun MuV_fun(data, para);
  CoxFM_updateBetaFun Beta_fun(data,  para);

  double ELBO = CoxFM_calcELBO(data, para);
  arma::vec ELBO_vec(maxiter);
  int iter;

  //double err_ELBO=0;
  double err_para;
  arma::vec para_prev = CoxFM_combinaPara(para);
  arma::vec para_after = para_prev;

  //Rcout <<  para_prev << "\n";

  for(iter=0; iter < maxiter; iter++){
    // Rcout << ELBO << "\n";
    para_prev = CoxFM_combinaPara(para);

    // update V and mu -- variational para
    //Rcout << "31\n";
    for(int i=0; i < data.n; i++){
      MuV_fun.i = i;
      //Rcout << "311\n";
      arma::vec mu = para.mu(i);
      arma::vec Lvec = para.Lvec(i);
      arma::vec muV(Lvec.n_elem + mu.n_elem);
      muV.subvec(0, data.p_z-1) = mu;
      muV.subvec(data.p_z, muV.n_elem-1) = Lvec;

      lbfgs.Optimize(MuV_fun,muV);
      //Rcout << "312\n";
      mu = muV.subvec(0, data.p_z-1);
      mu.replace(arma::datum::nan, 0);
      Lvec = muV.subvec(data.p_z, muV.n_elem-1);
      Lvec.replace(arma::datum::nan, 1);

      CoxFM_storeMuV(para,  mu, Lvec, i);

      //ELBO = CoxFM_calcELBO(data, para);
      //Rcout << ELBO << "\n";

    }

    //ELBO = CoxFM_calcELBO(data, para);
    //Rcout << ELBO << "\n";

    // update beta
    //Rcout << "32\n";
    arma::vec beta = para.beta;
    lbfgs.Optimize(Beta_fun, beta);
    para.beta = beta;

    //ELBO = CoxFM_calcELBO(data, para);
    //Rcout << ELBO << "\n";


    // update Sigma
    //Rcout << "33\n";
    CoxFM_updateSig(data,  para);
    //ELBO = CoxFM_calcELBO(data, para);

    para_after = CoxFM_combinaPara(para);

    ELBO_vec(iter) = ELBO;
    //Rcout << "iter="<< iter << "; EBLO=" << ELBO <<"\n";

    if(iter >= 0){
      // if(iter>0){
      //   err_ELBO = std::fabs(ELBO_vec(iter)-ELBO_vec(iter-1))/data.n;
      // }
      //double err_ELBO = (ELBO_vec(iter)-ELBO_vec(iter-1))/ELBO_vec(iter-1);
      err_para = std::sqrt(
        arma::accu(arma::square(para_after-para_prev))/para_prev.n_elem
      );
      // err_ELBO < eps or
      if( err_para<eps){
        break;
      }

    }

  }

  //Rcout <<"iters:" <<  iter << "; err_ELBO="<< err_ELBO << "; err_para=" << err_para <<"\n";


  return List::create(
    _["Sigma"] = para.Sigma,
    _["beta"] = para.beta,
    _["mu"] = para.mu,
    _["V"] = para.V,
    _["ELBO"] = ELBO_vec,
    _["iter"] = iter
  );
}



//// below are functions for PJFM /////

struct PJFM_data_t{
  // data part //
  arma::field<arma::mat> X; // n \times K mat - recur
  arma::field<arma::mat> Z; // n \times K mat - recur
  arma::field<arma::vec> Z_T; // n \times K vec - recur
  arma::field<arma::mat> X_t; // n \times K mat - recur
  arma::field<arma::mat> Z_t; // n \times K mat - recur
  arma::mat W_T; //matrix - surv
  arma::field<arma::mat> W_t; // n \times 1 mat - surv
  arma::field<arma::vec> GQ_w; // n \times 1 vec, Gauss-quadrature weights
  arma::field<arma::vec> GQ_t; // n \times 1 vec, Gauss-quadrature nodes

  // the below is for test data only //
  arma::field<arma::mat> Z_t_delta; // n \times K mat - recur for test data
  arma::field<arma::mat> W_t_delta; // n \times 1 mat - surv for test data
  arma::field<arma::vec> GQ_w_delta; // n \times 1 vec, Gauss-quadrature weights
  arma::field<arma::vec> GQ_t_delta; // n \times 1 vec, Gauss-quadrature nodes
  //////////////////////////////////////

  arma::vec ftime; // n \times 1 vec
  arma::vec fstat; //  n \times 1 vec
  arma::vec samWt; // n \times 1 vec: sampling weights

  int K; //total number of biomarkers
  int n; //total number of subjects
  int p_x; //total number of fixed-effects
  int p_z; //total number of random-effects
  arma::uvec p_x_vec; //number of fixed-effects for each biomarker
  arma::uvec p_z_vec; //number of random-effects for each biomarker
  arma::umat V_idx;

  // initialization function //
  PJFM_data_t(const List& datalist, const bool testData=false)
  {
    //Rcout << "data1\n";
    ftime = as<arma::vec> (datalist["ftime"]);
    fstat = as<arma::vec> (datalist["fstat"]);
    samWt = as<arma::vec> (datalist["samWt"]);
    n = ftime.n_elem;

    //Rcout << "data2\n";
    arma::field<arma::vec> GQ_w_tmp = datalist["GQ_w"];
    GQ_w = GQ_w_tmp;
    GQ_w_tmp.clear();
    arma::field<arma::vec> GQ_t_tmp = datalist["GQ_t"];
    GQ_t = GQ_t_tmp;
    GQ_t_tmp.clear();

    //Rcout << "data3\n";
    arma::field<arma::mat> X_tmp = datalist["X"];
    K = X_tmp.n_elem / n;
    X = arma::field<arma::mat>(n,K);
    field_reshape_mat(X_tmp, X,  n, K);
    X_tmp.clear();


    //Rcout << "data4\n";
    p_x_vec = arma::uvec(K);
    for(int k=0; k<K; k++){
      p_x_vec(k) = X(0,k).n_cols;
    }
    p_x = arma::accu(p_x_vec);

    arma::field<arma::mat> X_t_tmp = datalist["X_t"];
    X_t = arma::field<arma::mat>(n,K);
    field_reshape_mat(X_t_tmp, X_t,  n, K);
    X_t_tmp.clear();

    arma::field<arma::mat> Z_tmp = datalist["Z"];
    Z = arma::field<arma::mat>(n,K);
    field_reshape_mat(Z_tmp, Z,  n, K);
    Z_tmp.clear();

    //Rcout << "data5\n";
    V_idx = arma::umat(K,2);
    p_z_vec = arma::uvec(K);
    int start = 0;
    for(int k=0; k<K; k++){
      p_z_vec(k) = Z(0,k).n_cols;
      V_idx(k,0) = start;
      V_idx(k,1) = start + p_z_vec(k) - 1;
      start = start + p_z_vec(k);
    }
    p_z = arma::accu(p_z_vec);

    arma::field<arma::vec> Z_T_tmp = datalist["Z_T"];
    Z_T = arma::field<arma::vec>(n,K);
    field_reshape_vec(Z_T_tmp, Z_T,  n, K);
    Z_T_tmp.clear();

    //Rcout << "data6\n";
    arma::field<arma::mat> Z_t_tmp = datalist["Z_t"];
    Z_t = arma::field<arma::mat>(n,K);
    field_reshape_mat(Z_t_tmp, Z_t,  n, K);
    Z_t_tmp.clear();

    W_T = as<arma::mat>(datalist["W_T"]);

    //Rcout << "data7\n";
    arma::field<arma::mat> W_t_tmp = datalist["W_t"];
    W_t = W_t_tmp;
    W_t_tmp.clear();

    if(testData){
      arma::field<arma::mat> Z_t_tmp = datalist["Z_t_delta"];
      Z_t_delta = arma::field<arma::mat>(n,K);
      field_reshape_mat(Z_t_tmp, Z_t_delta,  n, K);
      Z_t_tmp.clear();

      arma::field<arma::mat> W_t_tmp = datalist["W_t_delta"];
      W_t_delta = W_t_tmp;
      W_t_tmp.clear();

      arma::field<arma::vec> GQ_w_tmp = datalist["GQ_w_delta"];
      GQ_w_delta = GQ_w_tmp;
      GQ_w_tmp.clear();

      arma::field<arma::vec> GQ_t_tmp = datalist["GQ_t_delta"];
      GQ_t_delta = GQ_t_tmp;
      GQ_t_tmp.clear();
    }

  }

};


// parameter struct
struct PJFM_para_t{
  // para part //
  arma::field<arma::vec> beta; // K \times 1 vec
  arma::mat Sigma; // dim = (q \times K)
  arma::mat invSigma; // inverse of Sigma
  arma::field<arma::vec> mu; // n \times K vec
  arma::field<arma::mat> V; // n \times 1 mat
  arma::field<arma::vec> Lvec; // n \times 1 vec: Lvec*Lvec.t() = V

  arma::vec beta0; // dim = p_w
  arma::vec alpha; // dim = K

  arma::uvec alpha_idx; // index for nonzero alphas
  int cov_nonzero=0; // nonzero alpha elements in inv_sigma

  // arma::uvec npara_vec; // num. of parameters in beta, gamma, alpha, weib
  // initialization function //
  PJFM_para_t(const List& paralist)
  {
    Sigma = as<arma::mat>(paralist["Sigma"]);
    beta0 = as<arma::vec>(paralist["beta0"]);
    alpha = as<arma::vec>(paralist["alpha"]);

    int K = alpha.n_elem;
    arma::field<arma::mat> V_tmp = paralist["V"];
    V = V_tmp;
    V_tmp.clear();
    int n = V.n_elem;
    Lvec = arma::field<arma::vec>(n);
    for(int i=0; i<n; i++){
      // Cholesky decomposition
      //arma::mat Ltmp = arma::chol(V(i),"lower");
      arma::mat Ltmp = myCholCpp(V(i));
      arma::uvec lower_indices = arma::trimatl_ind(arma::size(Ltmp));
      Lvec(i) = Ltmp(lower_indices);
    }

    arma::field<arma::vec> mu_tmp = paralist["mu"];
    mu = arma::field<arma::vec>(n,K);
    field_reshape_vec(mu_tmp, mu,  n, K);
    mu_tmp.clear();

    arma::field<arma::vec> beta_tmp = paralist["beta"];
    beta = beta_tmp;
    beta_tmp.clear();

    invSigma = myinvCpp(Sigma);

    // npara_vec = arma::uvec(4, arma::fill::zeros);
    // for(int k=0; k<beta.n_elem; k++){
    //   npara_vec(0) += beta(k).n_elem;
    // }
    // npara_vec(1) = gamma.n_elem;
    // npara_vec(2) = alpha.n_elem;
    // npara_vec(3) = weib.n_elem;
  }

  void updateInvSigma(){
    invSigma = myinvCpp(Sigma);
  }

  void NonZeroAlpha(){
    alpha_idx = arma::find(alpha);
  }

  void MakeDiagonal(){

    arma::vec tmp = invSigma.diag();
    invSigma.zeros();
    invSigma.diag() = tmp;

    tmp = Sigma.diag();
    Sigma.zeros();
    Sigma.diag() = tmp;

    for(int i=0; i<V.n_elem; i++){
      // Cholesky decomposition
      tmp = V(i).diag();
      V(i).zeros();
      V(i).diag() = tmp;
      Lvec(i) = arma::sqrt(tmp);
    }

  }

  void NonZeroInvSigma(){
    if(invSigma.n_rows > 1){
      arma::uvec upper_idx = arma::trimatu_ind( size(invSigma),  1); // upper-tri index
      arma::vec upper_part = invSigma(upper_idx);
      arma::uvec upper_nonzero = arma::find(upper_part);
      cov_nonzero = upper_nonzero.n_elem;
    }else{
      cov_nonzero = 0;
    }

  }

};


// calculate ELBO //
double PJFM_calcELBO(const PJFM_data_t& data,
                     const PJFM_para_t& para){

  //double ELBO=0;
  arma::vec ELBO(data.n, arma::fill::zeros);

  for(int i=0; i< data.n; i++){

    // recurrent
    for(int k=0; k<data.K; k++){
      ELBO(i) += arma::accu(data.X(i,k) * para.beta(k) +
        data.Z(i,k) * para.mu(i,k));

      arma::vec h_it = data.X_t(i,k) * para.beta(k) +
        data.Z_t(i,k) * para.mu(i,k);
      for(int j=0; j< h_it.n_elem; j++){
        h_it(j) += 0.5 * arma::as_scalar(
          data.Z_t(i,k).row(j) *
            para.V(i).submat(data.V_idx(k,0), data.V_idx(k,0),
                   data.V_idx(k,1), data.V_idx(k,1))*
                     data.Z_t(i,k).row(j).t()
        );
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);
    }

    // surv
    if(data.fstat(i) == 1){
      ELBO(i) +=  arma::as_scalar(data.W_T.row(i) * para.beta0);

      for(int k=0; k<data.K; k++){
        ELBO(i) +=  arma::accu(data.Z_T(i,k)%para.mu(i,k)) *para.alpha(k);
      }
    }

    arma::vec h_it =data.W_t(i) * para.beta0;
    arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, para.alpha,
                                               i, data.p_z_vec);
    for(int k=0; k<data.K; k++){
      h_it +=  data.Z_t(i,k)*para.mu(i,k)*para.alpha(k);
    }

    for(int j=0; j< h_it.n_elem; j++){
      h_it(j) += 0.5 * arma::as_scalar(
        Z_ia_t.row(j) * para.V(i) * Z_ia_t.row(j).t()
      );
    }

    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);

    ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

    double val, sign;
    arma::log_det(val, sign, para.Sigma);
    ELBO(i) -= 0.5 * val;

    arma::vec mu = field_to_vec(para.mu.row(i), data.p_z_vec);

    ELBO(i) -= 0.5 * arma::as_scalar(mu.t() * para.invSigma *mu);

    ELBO(i) -= 0.5 * arma::trace(para.invSigma * para.V(i));

    arma::log_det(val, sign, para.V(i));
    ELBO(i) += 0.5 * val;

  }

  return arma::accu(ELBO % data.samWt);
}

// update alpha//

class PJFM_updateAlphaFun{
public:
  const PJFM_data_t& data;
  const PJFM_para_t& para;

  PJFM_updateAlphaFun(const PJFM_data_t& data,
                      const PJFM_para_t& para):
    data(data), para(para){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& alpha_t, arma::mat& g)
  {

    arma::vec alpha = alpha_t.col(0);
    arma::mat grad_alpha(alpha.n_elem, data.n, arma::fill::zeros);
    arma::vec ELBO(data.n, arma::fill::zeros);

    for(int i=0; i< data.n; i++){

      // surv
      if(data.fstat(i) == 1){
        for(int k=0; k<data.K; k++){
          ELBO(i) +=  arma::accu(data.Z_T(i,k)%para.mu(i,k)) *alpha(k);
        }
      }

      arma::vec h_it =data.W_t(i) * para.beta0;
      arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, alpha,
                                            i, data.p_z_vec);
      for(int k=0; k<data.K; k++){
        h_it +=  data.Z_t(i,k)*para.mu(i,k)*alpha(k);
      }

      for(int j=0; j< h_it.n_elem; j++){
        h_it(j) += 0.5 * arma::as_scalar(
          Z_ia_t.row(j) * para.V(i) * Z_ia_t.row(j).t()
        );
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);

      ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

      // gradient of alpha //
      arma::vec grad_alpha_tmp(alpha.n_elem, arma::fill::zeros);
      for(int k=0; k < alpha.n_elem;k++){

        if(data.fstat(i)==1){
          grad_alpha_tmp(k) += arma::accu(data.Z_T(i,k)%para.mu(i,k));
        }
        arma::vec XBZmu =  data.Z_t(i,k)*para.mu(i,k);

        for(int j=0; j< XBZmu.n_elem; j++){
          XBZmu(j) +=  arma::as_scalar(
            data.Z_t(i,k).row(j) * para.V(i).rows(data.V_idx(k,0), data.V_idx(k,1)) *
              Z_ia_t.row(j).t()
          );
        }
        grad_alpha_tmp(k) -= arma::accu(data.GQ_w(i) % h_it % XBZmu);
      }

      grad_alpha.col(i) = grad_alpha_tmp;

    }

    double fval= -1*arma::accu(ELBO % data.samWt)/data.n;
    //g.col(0) = -1* arma::sum(grad_alpha,1)/data.n;
    g.col(0) = -1* grad_alpha * data.samWt/data.n;

    return fval;
  }

};


// update alpha with penalty//

class PJFM_updateAlpha_lasso_Fun{
public:
  const PJFM_data_t& data;
  const PJFM_para_t& para;

  double lam = 0;
  double ridge = 0;
  const arma::vec& gvec;

  PJFM_updateAlpha_lasso_Fun(const PJFM_data_t& data,
                      const PJFM_para_t& para,
                      const arma::vec& gvec):
    data(data), para(para), gvec(gvec){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& alpha_t, arma::mat& g)
  {

    arma::vec alpha = alpha_t.col(0);
    arma::mat grad_alpha(alpha.n_elem, data.n, arma::fill::zeros);
    arma::vec ELBO(data.n, arma::fill::zeros);

    for(int i=0; i< data.n; i++){

      // surv
      if(data.fstat(i) == 1){
        for(int k=0; k<data.K; k++){
          ELBO(i) +=  arma::accu(data.Z_T(i,k)%para.mu(i,k)) *alpha(k);
        }
      }

      arma::vec h_it =data.W_t(i) * para.beta0;
      arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, alpha,
                                                 i, data.p_z_vec);
      for(int k=0; k<data.K; k++){
        h_it +=  data.Z_t(i,k)*para.mu(i,k)*alpha(k);
      }

      for(int j=0; j< h_it.n_elem; j++){
        h_it(j) += 0.5 * arma::as_scalar(
          Z_ia_t.row(j) * para.V(i) * Z_ia_t.row(j).t()
        );
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);

      ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

      // gradient of alpha //
      arma::vec grad_alpha_tmp(alpha.n_elem, arma::fill::zeros);
      for(int k=0; k < alpha.n_elem;k++){

        if(data.fstat(i)==1){
          grad_alpha_tmp(k) += arma::accu(data.Z_T(i,k)%para.mu(i,k));
        }
        arma::vec XBZmu =  data.Z_t(i,k)*para.mu(i,k);

        for(int j=0; j< XBZmu.n_elem; j++){
          XBZmu(j) +=  arma::as_scalar(
            data.Z_t(i,k).row(j) * para.V(i).rows(data.V_idx(k,0), data.V_idx(k,1)) *
              Z_ia_t.row(j).t()
          );
        }
        grad_alpha_tmp(k) -= arma::accu(data.GQ_w(i) % h_it % XBZmu);
      }

      grad_alpha.col(i) = grad_alpha_tmp;

    }

    // double fval= -1*arma::accu(ELBO)/data.n;
    // g.col(0) = -1* arma::sum(grad_alpha,1)/data.n;

    // penalty part //
    double fval = -1 * arma::accu(ELBO % data.samWt) + 0.5*ridge*arma::accu(alpha%alpha) +
      lam* arma::accu(gvec % arma::abs(alpha));
    // g.col(0) = -1* arma::sum(grad_alpha,1) + ridge*alpha +
    //   lam * gvec % arma::sign(alpha);

    g.col(0) = -1*grad_alpha * data.samWt + ridge*alpha +
      lam * gvec % arma::sign(alpha);

    return fval;
  }

};

// update beta0 abd beta's //
class PJFM_updateBetaALLFun{
public:
  const PJFM_data_t& data;
  const PJFM_para_t& para;

  PJFM_updateBetaALLFun(const PJFM_data_t& data,
                        const PJFM_para_t& para):
    data(data), para(para){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& beta_all, arma::mat& g)
  {

    arma::vec beta0 = beta_all(arma::span(0,para.beta0.n_elem-1), 0);
    arma::vec beta = beta_all(arma::span(para.beta0.n_elem , beta_all.n_rows-1), 0);
    arma::field<arma::vec> beta_f = vec_to_field(beta, data.p_x_vec);

    arma::mat grad_beta0(beta0.n_elem, data.n, arma::fill::zeros);
    arma::field<arma::vec> grad_beta(data.K);
    for(int k=0; k< data.K; k++){
      grad_beta(k) = arma::vec(beta_f(k).n_elem, arma::fill::zeros);
    }

    arma::vec ELBO(data.n, arma::fill::zeros);

    for(int i=0; i< data.n; i++){

      // recurrent
      for(int k=0; k<data.K; k++){
        ELBO(i) += arma::accu(data.X(i,k) * beta_f(k));

        arma::vec h_it = data.X_t(i,k) * beta_f(k) +
          data.Z_t(i,k) * para.mu(i,k);
        for(int j=0; j< h_it.n_elem; j++){
          h_it(j) += 0.5 * arma::as_scalar(
            data.Z_t(i,k).row(j) *
              para.V(i).submat(data.V_idx(k,0), data.V_idx(k,0),
                     data.V_idx(k,1), data.V_idx(k,1))*
                       data.Z_t(i,k).row(j).t()
          );
        }

        h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
        h_it = arma::exp(h_it);
        ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

        // gradient of beta//
        grad_beta(k) += ( arma::sum(data.X(i,k).t(),1) -
          data.X_t(i,k).t() * (data.GQ_w(i) % h_it) ) * data.samWt(i);
      }

      // surv
      if(data.fstat(i) == 1){
        ELBO(i) +=  arma::as_scalar(data.W_T.row(i) * beta0);
      }
      arma::vec h_it = data.W_t(i) * beta0;
      arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, para.alpha,
                                            i, data.p_z_vec);
      for(int k=0; k<data.K; k++){
        h_it +=  data.Z_t(i,k)*para.mu(i,k)*para.alpha(k);
      }
      for(int j=0; j< h_it.n_elem; j++){
        h_it(j) += 0.5 * arma::as_scalar(
          Z_ia_t.row(j) * para.V(i) * Z_ia_t.row(j).t()
        );
      }
      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);

      ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

      // gradient of beta0//
      arma::vec grad_beta0_tmp(beta0.n_elem, arma::fill::zeros);

      if(data.fstat(i) == 1){
        grad_beta0_tmp += data.W_T.row(i).t();
      }
      grad_beta0_tmp -= data.W_t(i).t() * (data.GQ_w(i) % h_it);
      grad_beta0.col(i) = grad_beta0_tmp;
    }


    arma::vec grad_beta_vec = -1*field_to_vec(grad_beta, data.p_x_vec)/data.n;
    //arma::vec grad_beta0_vec =  -1*arma::sum(grad_beta0,1)/data.n;
    arma::vec grad_beta0_vec =  -1*grad_beta0 * data.samWt /data.n;
    double fval= -1*arma::accu(ELBO % data.samWt)/data.n;
    g.col(0) = arma::join_cols(grad_beta0_vec, grad_beta_vec);

    return fval;
  }

};


// update variational parameters mu_i and V_i //
class PJFM_updateMuVFun{
public:
  const PJFM_data_t& data;
  const PJFM_para_t& para;
  int i = 0;

  PJFM_updateMuVFun(const PJFM_data_t& data,
                    const PJFM_para_t& para) :
    data(data), para(para){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& muV, arma::mat& g)
  {
    arma::vec mu = muV(arma::span(0,data.p_z-1), 0);
    arma::vec Lvec = muV(arma::span(data.p_z, muV.n_rows-1), 0);

    arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);
    arma::mat L =  makeLowTriMat( para.V(i),  Lvec);
    arma::mat V = L*L.t();

    double val;
    double sign;


    /// fun value
    double fval = 0.0;
    arma::vec grad_mu(mu.n_rows, arma::fill::zeros);
    arma::mat grad_V(L.n_rows, L.n_rows, arma::fill::zeros);
    arma::mat V_tmp = V;
    V_tmp.zeros();

    // recurrent
    for(int k=0; k<data.K; k++){

      fval += arma::accu(data.Z(i,k) * mu_f(k));
      arma::vec h_it = data.X_t(i,k) * para.beta(k) +
        data.Z_t(i,k) * mu_f(k);
      for(int j=0; j< h_it.n_elem; j++){
        h_it(j) += 0.5 * arma::as_scalar(
          data.Z_t(i,k).row(j) *
            V.submat(data.V_idx(k,0), data.V_idx(k,0),
                     data.V_idx(k,1), data.V_idx(k,1))*
                       data.Z_t(i,k).row(j).t()
        );
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      fval -=  arma::accu(data.GQ_w(i) % h_it);

      grad_mu.subvec(data.V_idx(k,0), data.V_idx(k,1)) +=
        arma::sum(data.Z(i,k).t(), 1) - data.Z_t(i,k).t() * (data.GQ_w(i) % h_it);
      V_tmp.submat(data.V_idx(k,0), data.V_idx(k,0),
                   data.V_idx(k,1), data.V_idx(k,1)) =
                     data.Z_t(i,k).t() *
                     arma::diagmat((data.GQ_w(i) % h_it)) *
                     data.Z_t(i,k);
    }
    grad_V -= V_tmp*L;

    // surv
    if(data.fstat(i) == 1){
      for(int k=0; k<data.K; k++){
        if(para.alpha(k)!=0){
          fval +=  arma::accu(data.Z_T(i,k)%mu_f(k)) *para.alpha(k);
          grad_mu.subvec(data.V_idx(k,0), data.V_idx(k,1)) +=
            para.alpha(k) * data.Z_T(i,k);
        }
      }
    }

    arma::vec h_it = data.W_t(i) * para.beta0;
    arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, para.alpha,
                                          i, data.p_z_vec);
    for(int k=0; k<data.K; k++){
      if(para.alpha(k)!=0){
        h_it +=  data.Z_t(i,k)*mu_f(k)*para.alpha(k);
      }
    }

    for(int j=0; j< h_it.n_elem; j++){

      arma::vec Z_ia_t_j = Z_ia_t.row(j).t();
      arma::uvec Z_ia_t_idx = arma::find(Z_ia_t_j);
      if(Z_ia_t_idx.n_elem > 0){
        h_it(j) += 0.5 * arma::as_scalar(
          Z_ia_t_j.elem(Z_ia_t_idx).t() * V.submat(Z_ia_t_idx,Z_ia_t_idx) *
            Z_ia_t_j.elem(Z_ia_t_idx)
        );
      }
      // h_it(j) += 0.5 * arma::as_scalar(
      //   Z_ia_t.row(j) * V * Z_ia_t.row(j).t()
      // );
    }

    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);

    fval -=  arma::accu(data.GQ_w(i) % h_it);
    grad_mu -=  Z_ia_t.t() * (data.GQ_w(i) % h_it);
    grad_V -= Z_ia_t.t() * arma::diagmat((data.GQ_w(i) % h_it)) *
      Z_ia_t * L;

    // variational part
    arma::log_det(val, sign, V);
    fval += -0.5*arma::as_scalar(mu.t() * para.invSigma *mu) -
      0.5*arma::trace(para.invSigma*V) +  0.5 * val;
    grad_mu -= para.invSigma *mu;
    grad_V += arma::trans(arma::inv( arma::trimatl(L))) -
      para.invSigma*L;

    /// gradient

    fval = -1*fval;

    g(arma::span(0,data.p_z-1),0) = -grad_mu;

    g(arma::span(data.p_z,muV.n_rows-1),0) = -LowTriVec(grad_V);

    //Rcout << fval << "\n";

    return fval;
  }

};



// to put the new updates into para //
void PJFM_storeMuV(const PJFM_data_t& data, PJFM_para_t& para,
                   const arma::vec& mu,  const arma::vec& Lvec,
                   const int& i){

  para.Lvec(i) = Lvec;
  arma::mat L =  makeLowTriMat( para.V(i),  Lvec);
  para.V(i) = L*L.t();


  arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);
  for(int k=0; k<data.K; k++){
    para.mu(i,k) = mu_f(k);
  }
}

// update variational parameters mu_i and V_i //
class PJFM_updateMuVFun_Diag{
public:
  const PJFM_data_t& data;
  const PJFM_para_t& para;
  int i = 0;

  PJFM_updateMuVFun_Diag(const PJFM_data_t& data,
                         const PJFM_para_t& para) :
    data(data), para(para){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& muV, arma::mat& g)
  {
    arma::vec mu = muV(arma::span(0,data.p_z-1), 0);
    arma::vec Lvec = muV(arma::span(data.p_z, muV.n_rows-1), 0);

    arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);
    arma::mat L =  arma::mat(Lvec.n_elem, Lvec.n_elem);
    L.zeros();
    L.diag() = Lvec;

    arma::mat V = L*L.t();

    double val;
    double sign;


    /// fun value
    double fval = 0.0;
    arma::vec grad_mu(mu.n_rows, arma::fill::zeros);
    arma::mat grad_V(L.n_rows, L.n_rows, arma::fill::zeros);
    arma::mat V_tmp = V;
    V_tmp.zeros();

    //Rcout << "11\n";
    // recurrent
    for(int k=0; k<data.K; k++){

      fval += arma::accu(data.Z(i,k) * mu_f(k));
      arma::vec h_it = data.X_t(i,k) * para.beta(k) +
        data.Z_t(i,k) * mu_f(k);
      for(int j=0; j< h_it.n_elem; j++){
        h_it(j) += 0.5 * arma::as_scalar(
          data.Z_t(i,k).row(j) *
            V.submat(data.V_idx(k,0), data.V_idx(k,0),
                     data.V_idx(k,1), data.V_idx(k,1))*
                       data.Z_t(i,k).row(j).t()
        );
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      fval -=  arma::accu(data.GQ_w(i) % h_it);

      grad_mu.subvec(data.V_idx(k,0), data.V_idx(k,1)) +=
        arma::sum(data.Z(i,k).t(), 1) - data.Z_t(i,k).t() * (data.GQ_w(i) % h_it);
      V_tmp.submat(data.V_idx(k,0), data.V_idx(k,0),
                   data.V_idx(k,1), data.V_idx(k,1)) =
                     data.Z_t(i,k).t() *
                     arma::diagmat((data.GQ_w(i) % h_it)) *
                     data.Z_t(i,k);
    }
    grad_V -= V_tmp*L;

    //Rcout << "22\n";
    // surv
    if(data.fstat(i) == 1){
      for(int k=0; k<data.K; k++){
        if(para.alpha(k)!=0){
          fval +=  arma::accu(data.Z_T(i,k)%mu_f(k)) *para.alpha(k);
          grad_mu.subvec(data.V_idx(k,0), data.V_idx(k,1)) +=
            para.alpha(k) * data.Z_T(i,k);
        }
      }
    }

    arma::vec h_it = data.W_t(i) * para.beta0;
    arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, para.alpha,
                                               i, data.p_z_vec);
    for(int k=0; k<data.K; k++){
      if(para.alpha(k)!=0){
        h_it +=  data.Z_t(i,k)*mu_f(k)*para.alpha(k);
      }
    }

    for(int j=0; j< h_it.n_elem; j++){

      arma::vec Z_ia_t_j = Z_ia_t.row(j).t();
      arma::uvec Z_ia_t_idx = arma::find(Z_ia_t_j);
      if(Z_ia_t_idx.n_elem > 0){
        h_it(j) += 0.5 * arma::as_scalar(
          Z_ia_t_j.elem(Z_ia_t_idx).t() * V.submat(Z_ia_t_idx,Z_ia_t_idx) *
            Z_ia_t_j.elem(Z_ia_t_idx)
        );
      }
      // h_it(j) += 0.5 * arma::as_scalar(
      //   Z_ia_t.row(j) * V * Z_ia_t.row(j).t()
      // );
    }

    //Rcout << "33\n";
    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);

    fval -=  arma::accu(data.GQ_w(i) % h_it);
    grad_mu -=  Z_ia_t.t() * (data.GQ_w(i) % h_it);
    grad_V -= Z_ia_t.t() * arma::diagmat((data.GQ_w(i) % h_it)) *
      Z_ia_t * L;

    //Rcout << "44\n";
    // variational part
    arma::log_det(val, sign, V);
    fval += -0.5*arma::as_scalar(mu.t() * para.invSigma *mu) -
      0.5*arma::trace(para.invSigma*V) +  0.5 * val;
    grad_mu -= para.invSigma *mu;
    grad_V += arma::trans(arma::inv( arma::trimatl(L))) -
      para.invSigma*L;

    /// gradient
    //Rcout << "55\n";

    fval = -1*fval;

    g(arma::span(0,data.p_z-1),0) = -grad_mu;

    //g(arma::span(data.p_z,muV.n_rows-1),0) = -LowTriVec(grad_V);
    g(arma::span(data.p_z,muV.n_rows-1),0) = -grad_V.diag();

    //Rcout << fval << "\n";

    return fval;
  }

};


// to put the new updates into para //
void PJFM_storeMuV_Diag(const PJFM_data_t& data, PJFM_para_t& para,
                   const arma::vec& mu,  const arma::vec& Lvec,
                   const int& i){

  para.Lvec(i) = Lvec;

  arma::mat L =  arma::mat(Lvec.n_elem, Lvec.n_elem);
  L.zeros();
  L.diag() = Lvec;

  para.V(i) = L*L.t();


  arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);
  for(int k=0; k<data.K; k++){
    para.mu(i,k) = mu_f(k);
  }
}

// update  Sigma //
void PJFM_updateSig(const PJFM_data_t& data,
                    PJFM_para_t& para){

  arma::mat Sigma_tmp(arma::size(para.Sigma), arma::fill::zeros);

  for(int i=0; i< data.n; i++){
    arma::vec mu = field_to_vec(para.mu.row(i), data.p_z_vec);
    //Sigma_tmp += mu * mu.t() + para.V(i);
    Sigma_tmp += (mu * mu.t() + para.V(i)) * data.samWt(i);
  }

  //Sigma_tmp /= data.n;
  Sigma_tmp /= arma::accu(data.samWt);
  para.Sigma = Sigma_tmp;
  para.invSigma = myinvCpp(Sigma_tmp);
  para.NonZeroInvSigma();
}


// combine all parameters into a vector
arma::vec PJFM_combinaPara(const PJFM_data_t& data,
                           const PJFM_para_t& para){

  arma::vec beta_all = arma::join_cols(para.beta0, field_to_vec(para.beta, data.p_x_vec));
  arma::vec sig_vec = LowTriVec(para.Sigma);

  return arma::join_cols(para.alpha, beta_all, sig_vec);
}


// calculate hessian //
// D2(log(det|Sigma^-1|))(vech(Sigma^-1))
arma::mat D2logdet_Sigma(const arma::mat& Sigma){
  //// hessian on vech(invSigma) ////
  ///  needs info on Sigma ////
  int start = 0, p_mu = Sigma.n_cols,
    p_Sigma = (p_mu+1)*p_mu/2;

  arma::mat D_Sigma_Sigma(p_Sigma, p_Sigma, arma::fill::zeros);

  for(int j=0; j<p_mu; j++){ // column
    for(int i=j; i<p_mu; i++){ // row

      arma::mat tmp = Sigma.col(i) * Sigma.row(j);
      if(i!=j){
        tmp = tmp + tmp.t();
      }
      arma::vec Dtmp = tmp.diag();
      tmp *= -2.0;
      tmp.diag() += Dtmp;
      D_Sigma_Sigma.col(start) = LowTriVec(tmp);
      start ++;
    }
  }
  return D_Sigma_Sigma;
}

// D1(tr(SX))(vech(S))
arma::vec D1trace(const arma::mat& X){
  arma::mat tmp = X + X.t();
  tmp.diag() -= X.diag();
  return LowTriVec(tmp);
}

// D2(tr(Sigma*V))(vech(Sigma^-1)vech(V))
arma::mat D2_Sigma_V_fun(const arma::mat& Sigma){
  //// hessian on vech(invSigma) ////
  ///  needs info on Sigma ////
  int start = 0, p_mu = Sigma.n_cols,
    p_Sigma = (p_mu+1)*p_mu/2;

  arma::mat D_Sigma_V(p_Sigma, p_Sigma, arma::fill::zeros);
  D_Sigma_V.diag() -= 1.0;
  for(int j=0; j<p_mu; j++){ // column
    for(int i=j; i<p_mu; i++){ // row
      if(i==j){
        D_Sigma_V(start,start) = -0.5;
      }
      start ++;
    }
  }
  return D_Sigma_V;
}


// calcHessian
arma::mat calcHessian(const PJFM_data_t& data,
                      const PJFM_para_t& para){

  int p_beta = data.p_x;
  arma::uvec p_beta_vec = data.p_x_vec;
  int p_alpha = data.K;
  int p_beta0 = para.beta0.n_elem;
  int p_Sigma = (1+para.invSigma.n_cols)*para.invSigma.n_cols/2;

  int p_mu = data.p_z;
  arma::uvec p_mu_vec = data.p_z_vec;
  int p_V = (1+p_mu)*p_mu/2;

  // only interested in covariance of these parameters,
  // not interested in mu and V
  int p_all = p_beta+p_alpha+p_beta0+p_Sigma;
  arma::uvec p_all_vec(6);
  p_all_vec(0) = p_beta; p_all_vec(1) = p_alpha; p_all_vec(2) = p_beta0;
  p_all_vec(3) = p_Sigma;  p_all_vec(4) = p_mu; p_all_vec(5) = p_V;

  arma::umat var_pos(p_all_vec.n_elem,2);
  int start = 0, end=0;
  int start_z = 0, end_z=0;
  for(int j=0; j<p_all_vec.n_elem;j++){
    end = start + p_all_vec(j)-1;
    var_pos(j,0) = start;
    var_pos(j,1) = end;
    start = end+1;
  }

  //Rcout << var_pos << "\n";

  arma::mat H_11(p_all,p_all,arma::fill::zeros);

  //// on invSigma ////
  arma::mat D_Sigma_Sigma(p_Sigma, p_Sigma, arma::fill::zeros);

  D_Sigma_Sigma = D2logdet_Sigma(para.Sigma);
  D_Sigma_Sigma *= (arma::accu(data.samWt)/2.0);
  H_11(arma::span(var_pos(3,0), var_pos(3,1)) ,
       arma::span(var_pos(3,0), var_pos(3,1))) = D_Sigma_Sigma;

  // D_Sigma_V is the same for all V_i //
  arma::mat D_Sigma_V(p_Sigma, p_V, arma::fill::zeros);
  D_Sigma_V = D2_Sigma_V_fun(para.Sigma);

  //arma::mat H_12_all(p_all,p_mu+p_V,arma::fill::zeros);
  //arma::mat H_22_all(p_mu+p_V,p_mu+p_V,arma::fill::zeros);
  //Rcout << "1\n";

  for(int i=0; i< data.n; i++){

    //Rcout << "11\n";
    arma::mat H_12(p_all,p_mu+p_V,arma::fill::zeros);
    arma::mat H_22(p_mu+p_V,p_mu+p_V,arma::fill::zeros);

    // recurrent
    arma::mat h_it_recur(data.GQ_t(i).n_elem , data.K);
    for(int k=0; k<data.K; k++){

      h_it_recur.col(k) = data.X_t(i,k) * para.beta(k) +
        data.Z_t(i,k) * para.mu(i,k);
      for(int j=0; j< h_it_recur.n_rows; j++){
        h_it_recur(j,k) += 0.5 * arma::as_scalar(
          data.Z_t(i,k).row(j) *
            para.V(i).submat(data.V_idx(k,0), data.V_idx(k,0),
                   data.V_idx(k,1), data.V_idx(k,1))*
                     data.Z_t(i,k).row(j).t()
        );
      }
      h_it_recur = arma::clamp(h_it_recur, -MAX_EXP, MAX_EXP);
      h_it_recur = arma::exp(h_it_recur);
    }

    // surv
    arma::vec h_it =data.W_t(i) * para.beta0;
    arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, para.alpha,
                                               i, data.p_z_vec);
    for(int k=0; k<data.K; k++){
      h_it +=  data.Z_t(i,k)*para.mu(i,k)*para.alpha(k);
    }

    for(int j=0; j< h_it.n_elem; j++){
      h_it(j) += 0.5 * arma::as_scalar(
        Z_ia_t.row(j) * para.V(i) * Z_ia_t.row(j).t()
      );
    }

    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);


    //Rcout << "12\n";
    //Rcout << "beta\n";
    ///// beta part /////
    arma::mat D_beta_beta(p_beta,p_beta,arma::fill::zeros);
    start = 0, end=0;
    for(int k=0;k<data.K;k++){
      end = start+data.p_x_vec(k)-1;
      D_beta_beta.submat(start,start,end,end) -=
        data.X_t(i,k).t() * arma::diagmat(data.GQ_w(i) % h_it_recur.col(k)) *
        data.X_t(i,k);
      start = end + 1;
    }

    H_11(arma::span(var_pos(0,0), var_pos(0,1)) ,
         arma::span(var_pos(0,0), var_pos(0,1))) += D_beta_beta  * data.samWt(i);

    //Rcout << "17\n";
    // beta_mu
    arma::mat D_beta_mu(p_beta, p_mu, arma::fill::zeros);
    start_z = 0, end_z=0;
    start = 0, end=0;
    for(int k=0;k<data.K;k++){
      end = start+data.p_x_vec(k)-1;
      end_z = start_z+data.p_z_vec(k)-1;
      D_beta_mu.submat(start,start_z,end,end_z) -=
        data.X_t(i,k).t() * arma::diagmat(data.GQ_w(i) % h_it_recur.col(k)) *
        data.Z_t(i,k);
      start = end + 1;
      start_z = end_z + 1;
    }

    //Rcout << "18\n";
    // beta_V
    arma::mat D_beta_V(p_beta, p_V, arma::fill::zeros);
    arma::field<arma::mat> Z_ij0_t = arma::field<arma::mat>(data.K);
    for(int k=0; k<data.K; k++){
      Z_ij0_t(k) =  field_to_zero_noalpha_mat_full(data.Z_t, i,
              data.p_z_vec,k);
    }

    start = 0, end=0;
    for(int k=0;k<data.K;k++){
      end = start+data.p_x_vec(k)-1;

      for(int j=0; j<h_it.n_elem; j++){
        D_beta_V.rows(start,end) -= h_it_recur(j,k) * data.GQ_w(i)(j)*
          data.X_t(i,k).row(j).t() *
          D1trace( 0.5* Z_ij0_t(k).row(j).t() *  Z_ij0_t(k).row(j) ).t();
      }

      start = end + 1;
    }


    H_12(arma::span(var_pos(0,0), var_pos(0,1)) ,
         arma::span(0, p_mu-1) ) += D_beta_mu * data.samWt(i);
    H_12(arma::span(var_pos(0,0), var_pos(0,1)) ,
         arma::span(p_mu, p_V+p_mu-1) ) += D_beta_V * data.samWt(i);

    //Rcout << "alpha\n";
    //// alpha part ////
    // alpha_alpha + alpha_beta0
    arma::mat D_alpha_alpha(p_alpha, p_alpha, arma::fill::zeros);
    arma::mat D_alpha_beta0(p_alpha, p_beta0, arma::fill::zeros);

    arma::vec mu = field_to_vec(para.mu.row(i), data.p_z_vec);

    arma::field<arma::mat> Z_iD = field_to_field_Dmat(
      data.Z_t.row(i), data.p_z_vec);

    for(int j=0; j< Z_iD.n_elem; j++){
      arma::vec D1_h_alpha_vec =
        Z_iD(j).t()*mu +  Z_iD(j).t()*para.V(i)* Z_ia_t.row(j).t();

      D_alpha_alpha -=  h_it(j) * data.GQ_w(i)(j)*
        ( D1_h_alpha_vec*D1_h_alpha_vec.t() +
        Z_iD(j).t()*para.V(i)*Z_iD(j) ) ;

      D_alpha_beta0 -= h_it(j) * data.GQ_w(i)(j)* D1_h_alpha_vec *
        data.W_t(i).row(j) ;
    }


    H_11(arma::span(var_pos(1,0), var_pos(1,1)) ,
         arma::span(var_pos(1,0), var_pos(1,1))) += D_alpha_alpha * data.samWt(i);

    H_11(arma::span(var_pos(1,0), var_pos(1,1)) ,
         arma::span(var_pos(2,0), var_pos(2,1))) += D_alpha_beta0 * data.samWt(i);

    // alpha_mu

    arma::mat D_alpha_mu(p_alpha, p_mu, arma::fill::zeros);
    arma::mat Z_TD = field_to_Dmat(data.Z_T.row(i),  data.p_z_vec);
    if(data.fstat(i)==1){
      D_alpha_mu += Z_TD.t();
    }

    for(int j=0; j< Z_iD.n_elem; j++){
      arma::vec D1_h_alpha_vec =
        Z_iD(j).t()*mu +  Z_iD(j).t()*para.V(i)* Z_ia_t.row(j).t();
      D_alpha_mu -=  h_it(j) * data.GQ_w(i)(j)*
        ( Z_iD(j).t()  + D1_h_alpha_vec * Z_ia_t.row(j)) ;
    }


    // alpha_V
    arma::mat D_alpha_V(p_alpha, p_V, arma::fill::zeros);
    for(int j=0; j< Z_iD.n_elem; j++){
      arma::vec D1_h_alpha_vec =
        Z_iD(j).t()*mu +  Z_iD(j).t()*para.V(i)* Z_ia_t.row(j).t();
      D_alpha_V -= h_it(j) * data.GQ_w(i)(j)*
        (D1_h_alpha_vec *
        D1trace( 0.5* Z_ia_t.row(j).t()*Z_ia_t.row(j) ).t() ) ;

      for(int k=0; k<data.K; k++){
        D_alpha_V.row(k) -=  h_it(j) * data.GQ_w(i)(j)*(
          D1trace ( Z_ia_t.row(j).t() * Z_iD(j).col(k).t() ).t() );
      }
    }

    H_12(arma::span(var_pos(1,0), var_pos(1,1)) ,
         arma::span(0, p_mu-1) ) += D_alpha_mu * data.samWt(i);
    H_12(arma::span(var_pos(1,0), var_pos(1,1)) ,
         arma::span(p_mu, p_V+p_mu-1) ) += D_alpha_V * data.samWt(i);

    //Rcout << "beta0\n";
    //Rcout << "19\n";
    ///// beta0 part /////
    arma::mat D_beta0_beta0(p_beta0, p_beta0, arma::fill::zeros);
    D_beta0_beta0 -=  data.W_t(i).t() * arma::diagmat(data.GQ_w(i) % h_it) *
      data.W_t(i) ;

    H_11(arma::span(var_pos(2,0), var_pos(2,1)) ,
         arma::span(var_pos(2,0), var_pos(2,1))) += D_beta0_beta0 * data.samWt(i);

    // beta0_mu
    arma::mat D_beta0_mu(p_beta0, p_mu, arma::fill::zeros);
    for(int j=0; j<h_it.n_elem; j++){
      D_beta0_mu -= h_it(j) * data.GQ_w(i)(j)* data.W_t(i).row(j).t() *
        Z_ia_t.row(j);
    }

    // beta0_V
    arma::mat D_beta0_V(p_beta0, p_V, arma::fill::zeros);
    for(int j=0; j<h_it.n_elem; j++){
      D_beta0_V -= h_it(j) * data.GQ_w(i)(j)* data.W_t(i).row(j).t() *
        D1trace( 0.5* Z_ia_t.row(j).t() *  Z_ia_t.row(j)).t() ;
    }

    H_12(arma::span(var_pos(2,0), var_pos(2,1)) ,
         arma::span(0, p_mu-1) ) += D_beta0_mu * data.samWt(i);
    H_12(arma::span(var_pos(2,0), var_pos(2,1)) ,
         arma::span(p_mu, p_V+p_mu-1) ) += D_beta0_V * data.samWt(i);

    //Rcout << "Sigma\n";
    ///// Sigma part /////

    // D_Sigma_mu
    arma::mat D_Sigma_mu(p_Sigma, p_mu, arma::fill::zeros);

    for(int j=0; j<p_mu; j++){
      arma::mat mu0(p_mu,p_mu, arma::fill::zeros);
      mu0.row(j) += mu.t();
      mu0.col(j) += mu;
      arma::vec Dtmp = mu0.diag();
      mu0 *= 2.0;
      mu0.diag() -= Dtmp;
      D_Sigma_mu.col(j) = -0.5 * LowTriVec(mu0);
    }

    // D_Sigma_V
    H_12(arma::span(var_pos(3,0), var_pos(3,1)) ,
         arma::span(0, p_mu-1) ) += D_Sigma_mu * data.samWt(i);
    H_12(arma::span(var_pos(3,0), var_pos(3,1)) ,
         arma::span(p_mu, p_V+p_mu-1) ) += D_Sigma_V * data.samWt(i);

    //Rcout << "muV\n";
    ///// mu and V part /////

    // Rcout << "1\n";
    arma::mat D_mu_mu(p_mu, p_mu, arma::fill::zeros);
    arma::mat D_mu_V(p_mu, p_V, arma::fill::zeros);
    arma::mat D_V_V(p_V, p_V, arma::fill::zeros);

    start = 0, end = 0;
    for(int k=0; k<data.K; k++){
      end = start + data.p_z_vec(k) - 1;
      D_mu_mu.submat(start, start, end,end) -=
        data.Z_t(i,k).t() * arma::diagmat(data.GQ_w(i) % h_it_recur.col(k)) *
        data.Z_t(i,k) ;

      for(int j=0; j<h_it.n_elem; j++){
        D_mu_V.rows(start,end) -= h_it_recur(j,k) * data.GQ_w(i)(j)*
          data.Z_t(i,k).row(j).t() *
          D1trace( 0.5* Z_ij0_t(k).row(j).t() *  Z_ij0_t(k).row(j) ).t();
        D_V_V -= h_it_recur(j,k) * data.GQ_w(i)(j)*
          D1trace( 0.5* Z_ij0_t(k).row(j).t() *  Z_ij0_t(k).row(j) )*
          D1trace( 0.5* Z_ij0_t(k).row(j).t() *  Z_ij0_t(k).row(j) ).t();
      }

      start = end+1;
    }

    // Rcout << "2\n";
    for(int j=0; j< h_it.n_elem; j++){
      D_mu_mu -= h_it(j) * data.GQ_w(i)(j)* Z_ia_t.row(j).t()*
        Z_ia_t.row(j);
      D_mu_V -= h_it(j) * data.GQ_w(i)(j)* Z_ia_t.row(j).t()*
        D1trace( 0.5* Z_ia_t.row(j).t() *  Z_ia_t.row(j)).t();

      D_V_V -= h_it(j) * data.GQ_w(i)(j)*
        D1trace( 0.5* Z_ia_t.row(j).t() *  Z_ia_t.row(j))*
        D1trace( 0.5* Z_ia_t.row(j).t() *  Z_ia_t.row(j)).t();
    }
    D_mu_mu -= para.invSigma;

    arma::mat invV = myinvCpp(para.V(i));
    D_V_V += 0.5 * D2logdet_Sigma(invV);


    H_22(arma::span(0, p_mu-1) , arma::span(0, p_mu-1)) +=
      D_mu_mu * data.samWt(i);
    H_22(arma::span(0, p_mu-1), arma::span(p_mu, p_V+p_mu-1)) +=
      D_mu_V * data.samWt(i);
    H_22(arma::span(p_mu, p_V+p_mu-1) , arma::span(p_mu, p_V+p_mu-1)) +=
      D_V_V * data.samWt(i);

    H_22 = arma::symmatu(H_22);
    H_11 = arma::symmatu(H_11);

    H_22 = arma::inv(H_22);
    H_11-= H_12 * H_22* H_12.t();
    //break;
  }

  return H_11;
  // return List::create(
  //     _["H11"] =H_11,
  //     _["H12"] = H_12_all,
  //     _["H22"]=H_22_all
  // );
}

// calcHessian
arma::mat calcHessianSUB(const PJFM_data_t& data,
                      const PJFM_para_t& para){

  int p_beta = data.p_x;
  arma::uvec p_beta_vec = data.p_x_vec;
  int p_alpha = data.K;
  int p_beta0 = para.beta0.n_elem;
  int p_Sigma = (1+para.invSigma.n_cols)*para.invSigma.n_cols/2;

  int p_mu = data.p_z;
  arma::uvec p_mu_vec = data.p_z_vec;
  int p_V = (1+p_mu)*p_mu/2;

  // only interested in covariance of these parameters,
  // not interested in mu and V
  int p_all = p_beta+p_alpha+p_beta0+p_Sigma;
  arma::uvec p_all_vec(6);
  p_all_vec(0) = p_beta; p_all_vec(1) = p_alpha; p_all_vec(2) = p_beta0;
  p_all_vec(3) = p_Sigma;  p_all_vec(4) = p_mu; p_all_vec(5) = p_V;

  arma::umat var_pos(p_all_vec.n_elem,2);
  int start = 0, end=0;
  int start_z = 0, end_z=0;
  for(int j=0; j<p_all_vec.n_elem;j++){
    end = start + p_all_vec(j)-1;
    var_pos(j,0) = start;
    var_pos(j,1) = end;
    start = end+1;
  }

  //Rcout << var_pos << "\n";

  arma::mat H_11(p_all,p_all,arma::fill::zeros);

  //// on invSigma ////
  arma::mat D_Sigma_Sigma(p_Sigma, p_Sigma, arma::fill::zeros);

  D_Sigma_Sigma = D2logdet_Sigma(para.Sigma);
  D_Sigma_Sigma *= (arma::accu(data.samWt)/2.0);
  H_11(arma::span(var_pos(3,0), var_pos(3,1)) ,
       arma::span(var_pos(3,0), var_pos(3,1))) = D_Sigma_Sigma;

  // D_Sigma_V is the same for all V_i //
  arma::mat D_Sigma_V(p_Sigma, p_V, arma::fill::zeros);
  D_Sigma_V = D2_Sigma_V_fun(para.Sigma);

  //arma::mat H_12_all(p_all,p_mu+p_V,arma::fill::zeros);
  //arma::mat H_22_all(p_mu+p_V,p_mu+p_V,arma::fill::zeros);
  //Rcout << "1\n";

  for(int i=0; i< data.n; i++){

    //Rcout << "11\n";
    arma::mat H_12(p_all,p_mu+p_V,arma::fill::zeros);
    arma::mat H_22(p_mu+p_V,p_mu+p_V,arma::fill::zeros);

    // recurrent
    arma::mat h_it_recur(data.GQ_t(i).n_elem , data.K);
    for(int k=0; k<data.K; k++){

      h_it_recur.col(k) = data.X_t(i,k) * para.beta(k) +
        data.Z_t(i,k) * para.mu(i,k);
      for(int j=0; j< h_it_recur.n_rows; j++){
        h_it_recur(j,k) += 0.5 * arma::as_scalar(
          data.Z_t(i,k).row(j) *
            para.V(i).submat(data.V_idx(k,0), data.V_idx(k,0),
                   data.V_idx(k,1), data.V_idx(k,1))*
                     data.Z_t(i,k).row(j).t()
        );
      }
      h_it_recur = arma::clamp(h_it_recur, -MAX_EXP, MAX_EXP);
      h_it_recur = arma::exp(h_it_recur);
    }

    // surv
    arma::vec h_it =data.W_t(i) * para.beta0;
    arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, para.alpha,
                                               i, data.p_z_vec);
    for(int k=0; k<data.K; k++){
      h_it +=  data.Z_t(i,k)*para.mu(i,k)*para.alpha(k);
    }

    for(int j=0; j< h_it.n_elem; j++){
      h_it(j) += 0.5 * arma::as_scalar(
        Z_ia_t.row(j) * para.V(i) * Z_ia_t.row(j).t()
      );
    }

    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);


    //Rcout << "12\n";
    //Rcout << "beta\n";
    ///// beta part /////
    arma::mat D_beta_beta(p_beta,p_beta,arma::fill::zeros);
    start = 0, end=0;
    for(int k=0;k<data.K;k++){
      end = start+data.p_x_vec(k)-1;
      D_beta_beta.submat(start,start,end,end) -=
        data.X_t(i,k).t() * arma::diagmat(data.GQ_w(i) % h_it_recur.col(k)) *
        data.X_t(i,k);
      start = end + 1;
    }

    H_11(arma::span(var_pos(0,0), var_pos(0,1)) ,
         arma::span(var_pos(0,0), var_pos(0,1))) += D_beta_beta  * data.samWt(i);

    //Rcout << "17\n";
    // beta_mu
    arma::mat D_beta_mu(p_beta, p_mu, arma::fill::zeros);
    start_z = 0, end_z=0;
    start = 0, end=0;
    for(int k=0;k<data.K;k++){
      end = start+data.p_x_vec(k)-1;
      end_z = start_z+data.p_z_vec(k)-1;
      D_beta_mu.submat(start,start_z,end,end_z) -=
        data.X_t(i,k).t() * arma::diagmat(data.GQ_w(i) % h_it_recur.col(k)) *
        data.Z_t(i,k);
      start = end + 1;
      start_z = end_z + 1;
    }

    //Rcout << "18\n";
    // beta_V
    arma::mat D_beta_V(p_beta, p_V, arma::fill::zeros);
    arma::field<arma::mat> Z_ij0_t = arma::field<arma::mat>(data.K);
    for(int k=0; k<data.K; k++){
      Z_ij0_t(k) =  field_to_zero_noalpha_mat_full(data.Z_t, i,
              data.p_z_vec,k);
    }

    start = 0, end=0;
    for(int k=0;k<data.K;k++){
      end = start+data.p_x_vec(k)-1;

      for(int j=0; j<h_it.n_elem; j++){
        D_beta_V.rows(start,end) -= h_it_recur(j,k) * data.GQ_w(i)(j)*
          data.X_t(i,k).row(j).t() *
          D1trace( 0.5* Z_ij0_t(k).row(j).t() *  Z_ij0_t(k).row(j) ).t();
      }

      start = end + 1;
    }


    H_12(arma::span(var_pos(0,0), var_pos(0,1)) ,
         arma::span(0, p_mu-1) ) += D_beta_mu * data.samWt(i);
    H_12(arma::span(var_pos(0,0), var_pos(0,1)) ,
         arma::span(p_mu, p_V+p_mu-1) ) += D_beta_V * data.samWt(i);

    //Rcout << "alpha\n";
    //// alpha part ////
    // alpha_alpha + alpha_beta0
    arma::mat D_alpha_alpha(p_alpha, p_alpha, arma::fill::zeros);
    arma::mat D_alpha_beta0(p_alpha, p_beta0, arma::fill::zeros);

    arma::vec mu = field_to_vec(para.mu.row(i), data.p_z_vec);

    arma::field<arma::mat> Z_iD = field_to_field_Dmat(
      data.Z_t.row(i), data.p_z_vec);

    for(int j=0; j< Z_iD.n_elem; j++){
      arma::vec D1_h_alpha_vec =
        Z_iD(j).t()*mu +  Z_iD(j).t()*para.V(i)* Z_ia_t.row(j).t();

      D_alpha_alpha -=  h_it(j) * data.GQ_w(i)(j)*
        ( D1_h_alpha_vec*D1_h_alpha_vec.t() +
        Z_iD(j).t()*para.V(i)*Z_iD(j) ) ;

      D_alpha_beta0 -= h_it(j) * data.GQ_w(i)(j)* D1_h_alpha_vec *
        data.W_t(i).row(j) ;
    }


    H_11(arma::span(var_pos(1,0), var_pos(1,1)) ,
         arma::span(var_pos(1,0), var_pos(1,1))) += D_alpha_alpha * data.samWt(i);

    H_11(arma::span(var_pos(1,0), var_pos(1,1)) ,
         arma::span(var_pos(2,0), var_pos(2,1))) += D_alpha_beta0 * data.samWt(i);

    // alpha_mu

    arma::mat D_alpha_mu(p_alpha, p_mu, arma::fill::zeros);
    arma::mat Z_TD = field_to_Dmat(data.Z_T.row(i),  data.p_z_vec);
    if(data.fstat(i)==1){
      D_alpha_mu += Z_TD.t();
    }

    for(int j=0; j< Z_iD.n_elem; j++){
      arma::vec D1_h_alpha_vec =
        Z_iD(j).t()*mu +  Z_iD(j).t()*para.V(i)* Z_ia_t.row(j).t();
      D_alpha_mu -=  h_it(j) * data.GQ_w(i)(j)*
        ( Z_iD(j).t()  + D1_h_alpha_vec * Z_ia_t.row(j)) ;
    }


    // alpha_V
    arma::mat D_alpha_V(p_alpha, p_V, arma::fill::zeros);
    for(int j=0; j< Z_iD.n_elem; j++){
      arma::vec D1_h_alpha_vec =
        Z_iD(j).t()*mu +  Z_iD(j).t()*para.V(i)* Z_ia_t.row(j).t();
      D_alpha_V -= h_it(j) * data.GQ_w(i)(j)*
        (D1_h_alpha_vec *
        D1trace( 0.5* Z_ia_t.row(j).t()*Z_ia_t.row(j) ).t() ) ;

      for(int k=0; k<data.K; k++){
        D_alpha_V.row(k) -=  h_it(j) * data.GQ_w(i)(j)*(
          D1trace ( Z_ia_t.row(j).t() * Z_iD(j).col(k).t() ).t() );
      }
    }

    H_12(arma::span(var_pos(1,0), var_pos(1,1)) ,
         arma::span(0, p_mu-1) ) += D_alpha_mu * data.samWt(i);
    H_12(arma::span(var_pos(1,0), var_pos(1,1)) ,
         arma::span(p_mu, p_V+p_mu-1) ) += D_alpha_V * data.samWt(i);

    //Rcout << "beta0\n";
    //Rcout << "19\n";
    ///// beta0 part /////
    arma::mat D_beta0_beta0(p_beta0, p_beta0, arma::fill::zeros);
    D_beta0_beta0 -=  data.W_t(i).t() * arma::diagmat(data.GQ_w(i) % h_it) *
      data.W_t(i) ;

    H_11(arma::span(var_pos(2,0), var_pos(2,1)) ,
         arma::span(var_pos(2,0), var_pos(2,1))) += D_beta0_beta0 * data.samWt(i);

    // beta0_mu
    arma::mat D_beta0_mu(p_beta0, p_mu, arma::fill::zeros);
    for(int j=0; j<h_it.n_elem; j++){
      D_beta0_mu -= h_it(j) * data.GQ_w(i)(j)* data.W_t(i).row(j).t() *
        Z_ia_t.row(j);
    }

    // beta0_V
    arma::mat D_beta0_V(p_beta0, p_V, arma::fill::zeros);
    for(int j=0; j<h_it.n_elem; j++){
      D_beta0_V -= h_it(j) * data.GQ_w(i)(j)* data.W_t(i).row(j).t() *
        D1trace( 0.5* Z_ia_t.row(j).t() *  Z_ia_t.row(j)).t() ;
    }

    H_12(arma::span(var_pos(2,0), var_pos(2,1)) ,
         arma::span(0, p_mu-1) ) += D_beta0_mu * data.samWt(i);
    H_12(arma::span(var_pos(2,0), var_pos(2,1)) ,
         arma::span(p_mu, p_V+p_mu-1) ) += D_beta0_V * data.samWt(i);

    //Rcout << "Sigma\n";
    ///// Sigma part /////

    // D_Sigma_mu
    arma::mat D_Sigma_mu(p_Sigma, p_mu, arma::fill::zeros);

    for(int j=0; j<p_mu; j++){
      arma::mat mu0(p_mu,p_mu, arma::fill::zeros);
      mu0.row(j) += mu.t();
      mu0.col(j) += mu;
      arma::vec Dtmp = mu0.diag();
      mu0 *= 2.0;
      mu0.diag() -= Dtmp;
      D_Sigma_mu.col(j) = -0.5 * LowTriVec(mu0);
    }

    // D_Sigma_V
    H_12(arma::span(var_pos(3,0), var_pos(3,1)) ,
         arma::span(0, p_mu-1) ) += D_Sigma_mu * data.samWt(i);
    H_12(arma::span(var_pos(3,0), var_pos(3,1)) ,
         arma::span(p_mu, p_V+p_mu-1) ) += D_Sigma_V * data.samWt(i);

    //Rcout << "muV\n";
    ///// mu and V part /////

    // Rcout << "1\n";
    arma::mat D_mu_mu(p_mu, p_mu, arma::fill::zeros);
    arma::mat D_mu_V(p_mu, p_V, arma::fill::zeros);
    arma::mat D_V_V(p_V, p_V, arma::fill::zeros);

    start = 0, end = 0;
    for(int k=0; k<data.K; k++){
      end = start + data.p_z_vec(k) - 1;
      D_mu_mu.submat(start, start, end,end) -=
        data.Z_t(i,k).t() * arma::diagmat(data.GQ_w(i) % h_it_recur.col(k)) *
        data.Z_t(i,k) ;

      for(int j=0; j<h_it.n_elem; j++){
        D_mu_V.rows(start,end) -= h_it_recur(j,k) * data.GQ_w(i)(j)*
          data.Z_t(i,k).row(j).t() *
          D1trace( 0.5* Z_ij0_t(k).row(j).t() *  Z_ij0_t(k).row(j) ).t();
        D_V_V -= h_it_recur(j,k) * data.GQ_w(i)(j)*
          D1trace( 0.5* Z_ij0_t(k).row(j).t() *  Z_ij0_t(k).row(j) )*
          D1trace( 0.5* Z_ij0_t(k).row(j).t() *  Z_ij0_t(k).row(j) ).t();
      }

      start = end+1;
    }

    // Rcout << "2\n";
    for(int j=0; j< h_it.n_elem; j++){
      D_mu_mu -= h_it(j) * data.GQ_w(i)(j)* Z_ia_t.row(j).t()*
        Z_ia_t.row(j);
      D_mu_V -= h_it(j) * data.GQ_w(i)(j)* Z_ia_t.row(j).t()*
        D1trace( 0.5* Z_ia_t.row(j).t() *  Z_ia_t.row(j)).t();

      D_V_V -= h_it(j) * data.GQ_w(i)(j)*
        D1trace( 0.5* Z_ia_t.row(j).t() *  Z_ia_t.row(j))*
        D1trace( 0.5* Z_ia_t.row(j).t() *  Z_ia_t.row(j)).t();
    }
    D_mu_mu -= para.invSigma;

    arma::mat invV = myinvCpp(para.V(i));
    D_V_V += 0.5 * D2logdet_Sigma(invV);


    H_22(arma::span(0, p_mu-1) , arma::span(0, p_mu-1)) +=
      D_mu_mu * data.samWt(i);
    H_22(arma::span(0, p_mu-1), arma::span(p_mu, p_V+p_mu-1)) +=
      D_mu_V * data.samWt(i);
    H_22(arma::span(p_mu, p_V+p_mu-1) , arma::span(p_mu, p_V+p_mu-1)) +=
      D_V_V * data.samWt(i);

    H_22 = arma::symmatu(H_22);
    H_11 = arma::symmatu(H_11);

    H_22 = arma::inv(H_22);
    // H_11-= H_12 * H_22* H_12.t();
    //break;
  }

  return H_11;
  // return List::create(
  //     _["H11"] =H_11,
  //     _["H12"] = H_12_all,
  //     _["H22"]=H_22_all
  // );
}

//' Main function to run standard JFM
//' @noRd
//'
// [[Rcpp::export]]
List PJFM(const List& datalist, const List& paralist,
          int maxiter = 100, double eps=1e-4){

  PJFM_data_t data(datalist);
  PJFM_para_t para(paralist);

  double ELBO = PJFM_calcELBO(data, para);

  ens::L_BFGS lbfgs;
  lbfgs.MinGradientNorm() = MinGradientNorm;
  lbfgs.Factr() = Factr;
  PJFM_updateMuVFun MuV_fun(data, para);
  PJFM_updateBetaALLFun betaAll_fun(data,para);
  PJFM_updateAlphaFun alpha_fun(data,para);

  arma::vec ELBO_vec(maxiter);
  int iter;

  //double err_ELBO = 0;
  double err_para = 0;

  arma::vec para_prev = PJFM_combinaPara(data, para);
  arma::vec para_after = para_prev;

  for(iter=0; iter < maxiter; iter++){

    para_prev = PJFM_combinaPara(data, para);
    // update V and mu -- variational para
    //Rcout << "muv\n";
    for(int i=0; i < data.n; i++){
      MuV_fun.i = i;
      arma::vec mu = field_to_vec(para.mu.row(i), data.p_z_vec);
      arma::vec Lvec = para.Lvec(i);

      arma::vec muV(Lvec.n_elem + mu.n_elem);
      muV.subvec(0, data.p_z-1) = mu;
      muV.subvec(data.p_z, muV.n_elem-1) = Lvec;

      lbfgs.Optimize(MuV_fun,muV);

      mu = muV.subvec(0, data.p_z-1);
      Lvec = muV.subvec(data.p_z, muV.n_elem-1);
      PJFM_storeMuV(data, para,  mu, Lvec, i);

    }

    // ELBO = PJFM_calcELBO(data, para);
    // Rcout <<"var:" << ELBO/10000 << "\n";

    // update beta
    //Rcout << "beta\n";
    arma::vec beta_all(para.beta0.n_elem + data.p_x);
    beta_all.subvec(0, para.beta0.n_elem-1) = para.beta0;
    beta_all.subvec(para.beta0.n_elem, beta_all.n_elem-1) = field_to_vec(para.beta, data.p_x_vec);
    lbfgs.Optimize(betaAll_fun, beta_all);
    para.beta0 = beta_all.subvec(0, para.beta0.n_elem-1);
    para.beta  = vec_to_field(beta_all.subvec(para.beta0.n_elem, beta_all.n_elem-1), data.p_x_vec);

    // ELBO = PJFM_calcELBO(data, para);
    // Rcout <<"beta:" << ELBO/10000 << "\n";
    // Rcout << beta_all<< "\n";

    // update alpha
    //Rcout << "alpha\n";
    arma::vec alpha = para.alpha;
    lbfgs.Optimize(alpha_fun, alpha);
    para.alpha = alpha;
    //Rcout << alpha << "\n";

    // ELBO = PJFM_calcELBO(data, para);
    // Rcout <<"alpha:" << ELBO/10000 << "\n";

    // update sig
    //Rcout << "sig\n";
    PJFM_updateSig(data,  para);
    // ELBO = PJFM_calcELBO(data, para);
    // Rcout <<"sig:" << ELBO/10000 << "\n";

    ELBO = PJFM_calcELBO(data, para);
    ELBO_vec(iter) = ELBO;

    //Rcout << "iter="<< iter << "; EBLO=" << ELBO <<"\n";
    para_after = PJFM_combinaPara(data, para);

    if(iter >= 0){

      err_para = std::sqrt(
        arma::accu(arma::square(para_after-para_prev))/para_after.n_elem
      );
      // err_ELBO < eps or
      if( err_para<eps){
        break;
      }

    }

  }
  //
  //Rcout <<"iters:" <<  iter << "; err_ELBO="<< err_ELBO << "; err_para=" << err_para <<"\n";


  arma::mat H11 = calcHessian( data, para);
  arma::mat H11SUB = calcHessianSUB( data, para);

  return List::create(
    _["beta"] = para.beta,
    _["beta0"] = para.beta0,
    _["alpha"] = para.alpha,
    _["Sigma"] = para.Sigma,
    _["mu"] = para.mu,
    _["V"] = para.V,
    _["ELBO"] = ELBO_vec,
    _["iter"] = iter,
    _["H11"] = H11,
    _["H11SUB"] = H11SUB
  );
}



//' Main function to get hessian in PJFM
//' @noRd
//'
// [[Rcpp::export]]
double PJFM_numH(const arma::vec para_all, const List& datalist,
                 List& paralist, double eps=1e-4, bool noMUV=true){

  PJFM_data_t data(datalist);
  PJFM_para_t para(paralist);

  //
  int p_beta = data.p_x;
  arma::uvec p_beta_vec = data.p_x_vec;
  int p_beta0 = para.beta0.n_elem;
  int p_alpha = data.K;
  int p_Sigma = (1+para.invSigma.n_cols)*para.invSigma.n_cols/2;

  // only interested in covariance of these parameters,
  // not interested in mu and V
  //int p_all = p_beta+p_gamma+p_alpha+p_weib+p_Sigma+p_sig2;
  arma::uvec p_all_vec(4);
  p_all_vec(0) = p_beta; p_all_vec(1) = p_beta0;
  p_all_vec(2) = p_alpha;  p_all_vec(3) = p_Sigma;

  arma::field<arma::vec> para_field = vec_to_field(para_all, p_all_vec);
  arma::field<arma::vec> beta = vec_to_field(para_field(0), data.p_x_vec);
  para.beta = beta;
  para.beta0 = para_field(1);
  para.alpha = para_field(2);

  arma::mat L =  makeLowTriMat( para.Sigma,  para_field(3));
  para.Sigma = L*L.t();
  para.invSigma = myinvCpp(para.Sigma);

  if(!noMUV){
    //
    ens::L_BFGS lbfgs;
    lbfgs.MinGradientNorm() = MinGradientNorm;
    lbfgs.Factr() = Factr;

    PJFM_updateMuVFun MuV_fun(data, para);

    for(int i=0; i < data.n; i++){
      MuV_fun.i = i;
      arma::vec mu = field_to_vec(para.mu.row(i), data.p_z_vec);
      arma::vec Lvec = para.Lvec(i);

      arma::vec muV(Lvec.n_elem + mu.n_elem);
      muV.subvec(0, data.p_z-1) = mu;
      muV.subvec(data.p_z, muV.n_elem-1) = Lvec;

      lbfgs.Optimize(MuV_fun,muV);

      mu = muV.subvec(0, data.p_z-1);
      Lvec = muV.subvec(data.p_z, muV.n_elem-1);
      PJFM_storeMuV(data, para,  mu, Lvec, i);
    }
  }


  double ELBO = PJFM_calcELBO(data, para);
  return ELBO;
}



/// the following functions is for prediction ///

// update parameters b_i //
class PJFM_updateMuFun{
public:
  const PJFM_data_t& data;
  const PJFM_para_t& para;
  int i = 0;

  PJFM_updateMuFun(const PJFM_data_t& data,
                   const PJFM_para_t& para) :
    data(data), para(para){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& mu_t, arma::mat& g)
  {
    arma::vec mu = mu_t.col(0);
    arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);

    /// fun value
    double fval = 0.0;
    arma::vec grad_mu(mu.n_rows, arma::fill::zeros);

    // recurrent
    //Rcout << "recc\n";
    for(int k=0; k<data.K; k++){

      fval += arma::accu(data.Z(i,k) * mu_f(k));
      arma::vec h_it = data.X_t(i,k) * para.beta(k) +
        data.Z_t(i,k) * mu_f(k);

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      fval -=  arma::accu(data.GQ_w(i) % h_it);

      grad_mu.subvec(data.V_idx(k,0), data.V_idx(k,1)) +=
        arma::sum(data.Z(i,k).t(), 1) - data.Z_t(i,k).t() * (data.GQ_w(i) % h_it);
    }

    // surv
    //Rcout << "surv\n";
    arma::vec h_it = data.W_t(i) * para.beta0;
    arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, para.alpha,
                                               i, data.p_z_vec);
    for(int k=0; k<data.K; k++){
      h_it +=  data.Z_t(i,k)*mu_f(k)*para.alpha(k);
    }
    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);

    fval -=  arma::accu(data.GQ_w(i) % h_it);
    grad_mu -=  Z_ia_t.t() * (data.GQ_w(i) % h_it);

    // variational part
    //Rcout << "var\n";
    fval -= 0.5*arma::as_scalar(mu.t() * para.invSigma *mu);
    grad_mu -= para.invSigma *mu;

    /// gradient

    fval = -1*fval;
    g.col(0) = -grad_mu;

    return fval;
  }

};


// update parameters b_i //
class PJFM_updateMuDeltaFun{
public:
    const PJFM_data_t& data;
    const PJFM_para_t& para;
    int i = 0;

    PJFM_updateMuDeltaFun(const PJFM_data_t& data,
                     const PJFM_para_t& para) :
        data(data), para(para){
    }

    // Return the objective function with gradient.
    double EvaluateWithGradient(const arma::mat& mu_t, arma::mat& g)
    {
        arma::vec mu = mu_t.col(0);
        arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);

        /// fun value
        double fval = 0.0;
        arma::vec grad_mu(mu.n_rows, arma::fill::zeros);

        // recurrent
        //Rcout << "recc\n";
        for(int k=0; k<data.K; k++){

            fval += arma::accu(data.Z(i,k) * mu_f(k));
            arma::vec h_it = data.X_t(i,k) * para.beta(k) +
                data.Z_t(i,k) * mu_f(k);

            h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
            h_it = arma::exp(h_it);
            fval -=  arma::accu(data.GQ_w(i) % h_it);

            grad_mu.subvec(data.V_idx(k,0), data.V_idx(k,1)) +=
                arma::sum(data.Z(i,k).t(), 1) - data.Z_t(i,k).t() * (data.GQ_w(i) % h_it);
        }

        // surv
        //Rcout << "surv\n";
        arma::vec h_it = data.W_t_delta(i) * para.beta0;
        arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t_delta, para.alpha,
                                                   i, data.p_z_vec);
        for(int k=0; k<data.K; k++){
            h_it +=  data.Z_t_delta(i,k)*mu_f(k)*para.alpha(k);
        }
        h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
        h_it = arma::exp(h_it);

        fval -=  arma::accu(data.GQ_w_delta(i) % h_it);
        grad_mu -=  Z_ia_t.t() * (data.GQ_w_delta(i) % h_it);

        // variational part
        //Rcout << "var\n";
        fval -= 0.5*arma::as_scalar(mu.t() * para.invSigma *mu);
        grad_mu -= para.invSigma *mu;

        /// gradient

        fval = -1*fval;
        g.col(0) = -grad_mu;

        return fval;
    }

};

// function to eval log-density function in LP approx

double PJFM_log_surv(const PJFM_data_t& data,
                     const PJFM_para_t& para, const int i,
                     arma::vec mu){

    arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);

    /// fun value
    double fval = 0.0;
    // recurrent
    for(int k=0; k<data.K; k++){

        fval += arma::accu(data.X(i,k) * para.beta(k) +
          data.Z(i,k) * mu_f(k));
        arma::vec h_it = data.X_t(i,k) * para.beta(k) +
            data.Z_t(i,k) * mu_f(k);

        h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
        h_it = arma::exp(h_it);
        fval -=  arma::accu(data.GQ_w(i) % h_it);
    }

    // surv
    arma::vec h_it = data.W_t(i) * para.beta0;

    for(int k=0; k<data.K; k++){
        h_it +=  data.Z_t(i,k)*mu_f(k)*para.alpha(k);
    }
    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);

    fval -=  arma::accu(data.GQ_w(i) % h_it);

    // variational part
    fval -= 0.5*arma::as_scalar(mu.t() * para.invSigma *mu);

    return fval;
}

double PJFM_log_surv_delta(const PJFM_data_t& data,
                           const PJFM_para_t& para, const int i,
                           arma::vec mu){

    arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);

    /// fun value
    double fval = 0.0;

    // recurrent
    //Rcout << "recc\n";
    for(int k=0; k<data.K; k++){

        fval += arma::accu(data.X(i,k) * para.beta(k)+
          data.Z(i,k) * mu_f(k));

        arma::vec h_it = data.X_t(i,k) * para.beta(k) +
            data.Z_t(i,k) * mu_f(k);

        h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
        h_it = arma::exp(h_it);
        fval -=  arma::accu(data.GQ_w(i) % h_it);
    }

    // surv
    //Rcout << "surv\n";
    arma::vec h_it = data.W_t_delta(i) * para.beta0;

    for(int k=0; k<data.K; k++){
        h_it +=  data.Z_t_delta(i,k)*mu_f(k)*para.alpha(k);
    }
    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);

    fval -=  arma::accu(data.GQ_w_delta(i) % h_it);

    // variational part
    //Rcout << "var\n";
    fval -= 0.5*arma::as_scalar(mu.t() * para.invSigma *mu);

    return fval;
}

arma::mat PJFM_omega(const PJFM_data_t& data,
                     const PJFM_para_t& para, const int i,
                     arma::vec mu){

    arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);

    // calculate omega
    arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, para.alpha,
                                               i, data.p_z_vec);
    arma::field<arma::mat> Z_ij0_t = arma::field<arma::mat>(data.K);
    for(int k=0; k<data.K; k++){
        Z_ij0_t(k) =  field_to_zero_mat_full(data.Z_t, para.alpha,i,
                data.p_z_vec,k);
    }
    //
    arma::mat omega = para.invSigma;

    //Rcout << omega << "\n";
    // recur part
    arma::field<arma::vec> h_it_r(data.K);
    for(int k=0; k<data.K; k++){
        h_it_r(k) = data.X_t(i,k) * para.beta(k) + data.Z_t(i,k) * mu_f(k);
        h_it_r(k) = arma::clamp( h_it_r(k), -MAX_EXP, MAX_EXP);
        h_it_r(k) = arma::exp( h_it_r(k));
        omega += Z_ij0_t(k).t() * arma::diagmat(data.GQ_w(i) % h_it_r(k)) * Z_ij0_t(k);
    }
    //Rcout << omega << "\n";

    // surv part
    arma::vec h_it_s = data.W_t(i) * para.beta0;

    for(int k=0; k<data.K; k++){
        h_it_s +=  data.Z_t(i,k)*mu_f(k)*para.alpha(k);
    }

    h_it_s = arma::clamp(h_it_s, -MAX_EXP, MAX_EXP);
    h_it_s = arma::exp(h_it_s);
    omega += Z_ia_t.t() * arma::diagmat(data.GQ_w(i) % h_it_s) * Z_ia_t;

    //Rcout << omega << "\n";

    return omega;
}

arma::mat PJFM_omega_detla(const PJFM_data_t& data,
                           const PJFM_para_t& para, const int i,
                           arma::vec mu){

    arma::field<arma::vec> mu_f = vec_to_field(mu, data.p_z_vec);

    // calculate omega
    arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t_delta, para.alpha,
                                               i, data.p_z_vec);
    arma::field<arma::mat> Z_ij0_t = arma::field<arma::mat>(data.K);
    for(int k=0; k<data.K; k++){
        Z_ij0_t(k) =  field_to_zero_mat_full(data.Z_t, para.alpha,i,
                data.p_z_vec,k);
    }
    //
    arma::mat omega = para.invSigma;

    // recur part
    arma::field<arma::vec> h_it_r(data.K);
    for(int k=0; k<data.K; k++){
        h_it_r(k) = data.X_t(i,k) * para.beta(k) + data.Z_t(i,k) * mu_f(k);
        h_it_r(k) = arma::clamp( h_it_r(k), -MAX_EXP, MAX_EXP);
        h_it_r(k) = arma::exp( h_it_r(k));
        omega += Z_ij0_t(k).t() * arma::diagmat(data.GQ_w(i) % h_it_r(k)) * Z_ij0_t(k);
    }

    // surv part
    arma::vec h_it_s = data.W_t_delta(i) * para.beta0;

    for(int k=0; k<data.K; k++){
        h_it_s +=  data.Z_t_delta(i,k)*mu_f(k)*para.alpha(k);
    }

    h_it_s = arma::clamp(h_it_s, -MAX_EXP, MAX_EXP);
    h_it_s = arma::exp(h_it_s);
    omega += Z_ia_t.t() * arma::diagmat(data.GQ_w_delta(i) % h_it_s) * Z_ia_t;

    return omega;
}

//' Main function to calcuate the predicted scores for JFM
//' @noRd
//'
// [[Rcpp::export]]
arma::vec PJFM_pred(const List& datalist, const List& paralist){


    PJFM_data_t data(datalist, true);
    PJFM_para_t para(paralist);

    ens::L_BFGS lbfgs;
    //lbfgs.MinGradientNorm() = MinGradientNorm;
    //lbfgs.Factr() = Factr;
    PJFM_updateMuFun Mu_fun(data, para);
    PJFM_updateMuDeltaFun Mu_Delta_fun(data, para);

    arma::vec pred_scores(data.n);
    double val, sign, val_delta;

    //Rcout << data.n << "\n";
    for(int i=0; i < data.n; i++){
        Mu_fun.i = i;
        Mu_Delta_fun.i = i;

        arma::vec mu = field_to_vec(para.mu.row(i), data.p_z_vec);
        lbfgs.Optimize(Mu_fun, mu);

        arma::vec mu_delta = field_to_vec(para.mu.row(i), data.p_z_vec);
        lbfgs.Optimize(Mu_Delta_fun, mu_delta);

        arma::mat omega = PJFM_omega( data, para,  i,  mu);
        arma::mat omega_delta = PJFM_omega_detla( data, para,  i,  mu_delta);

        arma::log_det(val, sign, omega);
        arma::log_det(val_delta, sign, omega_delta);

        double logl = PJFM_log_surv(data, para,  i,  mu);
        double logl_delta = PJFM_log_surv_delta(data, para,  i,  mu_delta);

        //Rcout << i << "\n";
        pred_scores(i) = 1.0 - std::exp(logl_delta - 0.5*val_delta - logl +
            0.5*val);
        //Rcout << ":"<< i+1 << "\n";
    }

    return pred_scores;

}


///// the following functions for lasso PJFM ////

// parameter struct
struct PJFM_para_covBD_t{
  // para part //
  arma::field<arma::vec> beta; // K \times 1 vec

  arma::field<arma::mat> Sigma; // K \times 1 mat
  arma::field<arma::mat> invSigma; // inverse of Sigma

  arma::field<arma::vec> mu; // n \times K vec
  arma::field<arma::mat> V; // n \times K mat
  arma::field<arma::vec> Lvec; // n \times K vec: Lvec*Lvec.t() = V

  arma::vec beta0; // dim = p_w
  arma::vec alpha; // dim = K

  arma::uvec alpha_idx; // index for nonzero alphas
  int p_x_alpha; // total number of fixed-effects for nonzero alphas
  int p_z_alpha; // total number of random-effects for nonzero alphas
  int p_zz_alpha; // total length of Lvec for nonzero alphas
  arma::uvec p_x_vec_alpha; //number of fixed-effects for each nonzero alpha
  arma::uvec p_z_vec_alpha; //number of random-effects for each nonzero alpha
  arma::uvec p_zz_vec_alpha; //length of Lvec for each nonzero alpha

  arma::uvec npara_vec; // num. of parameters in beta, gamma


  // arma::uvec npara_vec; // num. of parameters in beta, gamma, alpha, weib
  // initialization function //
  PJFM_para_covBD_t(const List& paralist)
  {
    beta0 = as<arma::vec>(paralist["beta0"]);
    alpha = as<arma::vec>(paralist["alpha"]);

    int K = alpha.n_elem;
    arma::field<arma::mat> V_tmp = paralist["V"];
    int n = V_tmp.n_elem/K;

    V = arma::field<arma::mat>(n,K);
    field_reshape_mat(V_tmp, V,  n, K);
    V_tmp.clear();

    Lvec = arma::field<arma::vec>(n,K);
    for(int i=0; i<n; i++){
      // Cholesky decomposition
      for(int k=0; k<K; k++){
        arma::mat Ltmp = arma::chol(V(i,k),"lower");
        arma::uvec lower_indices = arma::trimatl_ind(arma::size(Ltmp));
        Lvec(i,k) = Ltmp(lower_indices);
      }
    }

    arma::field<arma::vec> mu_tmp = paralist["mu"];
    mu = arma::field<arma::vec>(n,K);
    field_reshape_vec(mu_tmp, mu,  n, K);
    mu_tmp.clear();

    arma::field<arma::vec> beta_tmp = paralist["beta"];
    beta = beta_tmp;
    beta_tmp.clear();

    arma::field<arma::mat> Sigma_tmp = paralist["Sigma"];
    Sigma = Sigma_tmp;
    Sigma_tmp.clear();

    invSigma = arma::field<arma::mat>(K);
    for(int k=0; k<K; k++){
      invSigma(k) = myinvCpp(Sigma(k));
    }
    npara_vec = arma::uvec(2, arma::fill::zeros);
    for(int k=0; k<beta.n_elem; k++){
      npara_vec(0) += beta(k).n_elem;
    }
    npara_vec(1) = beta0.n_elem;
  }



  void NonZeroAlpha(){
    alpha_idx = arma::find(alpha);
    if(alpha_idx.n_elem > 0){
      p_z_vec_alpha = arma::uvec(alpha_idx.n_elem);
      p_zz_vec_alpha = arma::uvec(alpha_idx.n_elem);
      p_x_vec_alpha = arma::uvec(alpha_idx.n_elem);
      for(int j=0; j<alpha_idx.n_elem; j++){
        p_z_vec_alpha(j) = mu(0,alpha_idx(j)).n_elem;
        p_zz_vec_alpha(j) = p_z_vec_alpha(j)*(p_z_vec_alpha(j)+1)/2;
        p_x_vec_alpha(j) = beta(alpha_idx(j)).n_elem;
      }
      p_z_alpha = arma::accu(p_z_vec_alpha);
      p_zz_alpha = arma::accu(p_zz_vec_alpha);
      p_x_alpha = arma::accu(p_x_vec_alpha);

      npara_vec(0) = p_x_alpha;
    }
  }


};


// calculate baseline ELBO for recurrent events data  with zero alphas //
arma::mat PJFM_calcELBO_baseRecur(const PJFM_data_t& data,
                                  const PJFM_para_covBD_t& para){

  //double ELBO=0;
  arma::mat ELBO(data.n, data.K, arma::fill::zeros);

  for(int i=0; i< data.n; i++){

    // recurrent
    for(int k=0; k<data.K; k++){
      ELBO(i,k) += arma::accu(data.X(i,k) * para.beta(k) +
        data.Z(i,k) * para.mu(i,k));

      arma::vec h_it = data.X_t(i,k) * para.beta(k) +
        data.Z_t(i,k) * para.mu(i,k);
      for(int j=0; j< h_it.n_elem; j++){
        h_it(j) += 0.5 * arma::as_scalar(
          data.Z_t(i,k).row(j) * para.V(i,k)*
            data.Z_t(i,k).row(j).t()
        );
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      ELBO(i,k) -=  arma::accu(data.GQ_w(i) % h_it);
    }

    double val, sign;
    for(int k=0; k< data.K; k++){

      arma::log_det(val, sign, para.Sigma(k));
      ELBO(i,k) -= 0.5 * val;

      ELBO(i,k) -= 0.5 * arma::as_scalar(para.mu(i,k).t()*para.invSigma(k)*
        para.mu(i,k));
      ELBO(i,k) -= 0.5 * arma::trace(para.invSigma(k) * para.V(i,k));

      arma::log_det(val, sign, para.V(i,k));
      ELBO(i,k) += 0.5 * val;
    }
  }

  return ELBO;
}


// calculate ELBO //
double PJFM_calcELBO_covBD(const PJFM_data_t& data,
                           const PJFM_para_covBD_t& para,
                           const arma::mat& baseELBO,
                           const arma::vec&  gvec,
                           double lam, double ridge){

  //double ELBO=0;
  arma::mat ELBO = baseELBO;
  arma::vec ELBO_surv(data.n,arma::fill::zeros);

  int k = 0;
  double val, sign;

  // reset the columns for nonzero alpha_k
  for(int j=0; j< para.alpha_idx.n_elem; j++){
    k = para.alpha_idx(j);
    ELBO.col(k).zeros();
  }

  for(int i=0; i< data.n; i++){

    // Recurrent part
    for(int j=0; j< para.alpha_idx.n_elem; j++){

      k = para.alpha_idx(j);
      ELBO(i,k) += arma::accu(data.X(i,k) * para.beta(k) +
        data.Z(i,k) * para.mu(i,k));

      arma::vec h_it = data.X_t(i,k) * para.beta(k) +
        data.Z_t(i,k) * para.mu(i,k);
      for(int j=0; j< h_it.n_elem; j++){
        h_it(j) += 0.5 * arma::as_scalar(
          data.Z_t(i,k).row(j) * para.V(i,k)*
            data.Z_t(i,k).row(j).t()
        );
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      ELBO(i,k) -=  arma::accu(data.GQ_w(i) % h_it);

      arma::log_det(val, sign, para.Sigma(k));
      ELBO(i,k) -= 0.5 * val;

      ELBO(i,k) -= 0.5 * arma::as_scalar(para.mu(i,k).t()*para.invSigma(k)*
        para.mu(i,k));
      ELBO(i,k) -= 0.5 * arma::trace(para.invSigma(k) * para.V(i,k));

      arma::log_det(val, sign, para.V(i,k));
      ELBO(i,k) += 0.5 * val;
    }

    // Survival part
    if(data.fstat(i) == 1){
      ELBO_surv(i) += arma::as_scalar(data.W_T.row(i) * para.beta0);

      for(int j=0; j< para.alpha_idx.n_elem; j++){
        k = para.alpha_idx(j);
        ELBO_surv(i) += arma::accu(data.Z_T(i,k)%para.mu(i,k)) *para.alpha(k);
      }
    }

    arma::vec h_it = data.W_t(i) * para.beta0;

    for(int j=0; j< para.alpha_idx.n_elem; j++){
      k = para.alpha_idx(j);
      h_it += data.Z_t(i,k)*para.mu(i,k)*para.alpha(k);
    }

    for(int jj=0; jj < h_it.n_elem; jj++){
      for(int j=0; j<para.alpha_idx.n_elem; j++){
        k = para.alpha_idx(j);
        h_it(jj) +=  0.5 * para.alpha(k)*para.alpha(k) * arma::as_scalar(
          data.Z_t(i, k).row(jj) *  para.V(i,k)*
            data.Z_t(i, k).row(jj).t()
        );
      }
    }

    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);

    ELBO_surv(i) -=  arma::accu(data.GQ_w(i) % h_it);
  }

  double res = arma::accu(ELBO.t() * data.samWt) + arma::accu(ELBO_surv % data.samWt) -
    lam*arma::accu(gvec%arma::abs(para.alpha)) -
    0.5*ridge*arma::accu(arma::square(para.alpha));
  //res = res/data.n/data.K;

  return res;
  // return arma::accu(ELBO) + arma::accu(ELBO_surv) -
  //     lam*arma::accu(gvec%arma::abs(para.alpha)) -
  //     0.5*ridge*arma::accu(arma::square(para.alpha));
}

// get the maximum penalty values //
double PJFM_get_lammax_covBD(const PJFM_data_t& data,
                             const PJFM_para_covBD_t& para,
                             const arma::vec& gvec){

  arma::mat h_t = arma::mat(data.GQ_t(0).n_elem,data.n, arma::fill::zeros);
  arma::mat XBZmu_T = arma::mat(data.n, data.K, arma::fill::zeros);
  arma::field<arma::vec> XBZmu_t = arma::field<arma::vec>(data.n, data.K);
  arma::field<arma::vec> ZVZ = arma::field<arma::vec>(data.n, data.K);
  arma::vec tmp(data.GQ_t(0).n_elem, arma::fill::zeros);

  for(int i=0; i< data.n; i++){
    for(int k=0; k<data.K;k++){
      XBZmu_T(i,k) =  arma::accu(data.Z_T(i,k)%para.mu(i,k)) ;
      XBZmu_t(i,k) =  data.Z_t(i,k)*para.mu(i,k);
      for(int jj=0; jj<tmp.n_elem; jj++){

        tmp(jj) = arma::as_scalar(
          data.Z_t(i, k).row(jj) *  para.V(i,k)*
            data.Z_t(i, k).row(jj).t()
        );
      }
      ZVZ(i,k) = tmp;
    }
  }

  int k=0;

  for(int i=0; i< data.n; i++){
    arma::vec h_it = data.W_t(i) * para.beta0;
    for(int j=0; j< para.alpha_idx.n_elem; j++){
      k = para.alpha_idx(j);
      h_it += XBZmu_t(i,k) * para.alpha(k) +
        0.5 * para.alpha(k)*para.alpha(k) * ZVZ(i,k);
    }
    h_t.col(i) = h_it;
  }


  arma::vec grad(data.K,arma::fill::zeros);
  for(int k=0;k<data.K;k++){
    for(int i=0; i<data.n; i++){
      if(data.fstat(i)==1){
        grad(k) +=  data.samWt(i) * XBZmu_T(i,k);
      }
      arma::vec h_it = h_t.col(i);
      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      grad(k) -= data.samWt(i) * arma::accu(data.GQ_w(i) % h_it % XBZmu_t(i,k));
    }
  }

  return  arma::abs(grad/gvec).max()*1.2;
}

// update alpha//

class PJFM_updateAlphaVecLasso_covBD_Fun{
public:
  const PJFM_data_t& data;
  const PJFM_para_covBD_t& para;
  double lam = 0;
  double ridge = 0;
  const arma::vec& gvec;

  PJFM_updateAlphaVecLasso_covBD_Fun(const PJFM_data_t& data,
                                     const PJFM_para_covBD_t& para,
                                     const arma::vec& gvec):
    data(data), para(para),gvec(gvec){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& alpha_t, arma::mat& g)
  {

    arma::vec alpha = alpha_t.col(0);
    arma::vec gvec_sub = gvec.elem(para.alpha_idx);

    arma::mat grad_alpha(alpha.n_elem, data.n, arma::fill::zeros);
    arma::vec ELBO(data.n, arma::fill::zeros);
    int k=0;
    for(int i=0; i< data.n; i++){

      // surv
      if(data.fstat(i) == 1){
        // for(int k=0; k<data.K; k++){
        //   ELBO(i) +=  arma::accu(data.Z_T(i,k)%para.mu(i,k)) *alpha(k);
        // }
        for(int j=0; j< para.alpha_idx.n_elem; j++){
          k = para.alpha_idx(j);
          ELBO(i) += arma::accu(data.Z_T(i,k)%para.mu(i,k)) *alpha(j);
        }
      }

      arma::vec h_it = data.W_t(i) * para.beta0;
      // arma::mat Z_ia_t = field_to_alpha_mat_full(data.Z_t, alpha,
      //                                            i, data.p_z_vec);
      // for(int k=0; k<data.K; k++){
      //   h_it +=  data.Z_t(i,k)*para.mu(i,k)*alpha(k);
      // }
      //
      // for(int j=0; j< h_it.n_elem; j++){
      //   h_it(j) += 0.5 * arma::as_scalar(
      //     Z_ia_t.row(j) * para.V(i) * Z_ia_t.row(j).t()
      //   );
      // }
      //
      // h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      // h_it = arma::exp(h_it);

      for(int j=0; j< para.alpha_idx.n_elem; j++){
        k = para.alpha_idx(j);
        h_it += data.Z_t(i,k)*para.mu(i,k) * alpha(j);
      }

      for(int jj=0; jj < h_it.n_elem; jj++){
        for(int j=0; j<para.alpha_idx.n_elem; j++){
          k = para.alpha_idx(j);
          h_it(jj) +=  0.5 *alpha(j)*alpha(j) * arma::as_scalar(
            data.Z_t(i, k).row(jj) *  para.V(i,k)*
              data.Z_t(i, k).row(jj).t()
          );
        }
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);

      ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

      // gradient of alpha //
      arma::vec grad_alpha_tmp(alpha.n_elem, arma::fill::zeros);
      for(int j=0; j < alpha.n_elem;j++){
        k = para.alpha_idx(j);
        if(data.fstat(i)==1){
          grad_alpha_tmp(j) += arma::accu(data.Z_T(i,k)%para.mu(i,k));
        }
        arma::vec XBZmu =  data.Z_t(i,k)*para.mu(i,k);

        for(int jj=0; jj< XBZmu.n_elem; jj++){
          XBZmu(jj) +=  alpha(j) * arma::as_scalar(
            data.Z_t(i, k).row(jj) *  para.V(i,k)*
              data.Z_t(i, k).row(jj).t()
          );
        }

        grad_alpha_tmp(j) -= arma::accu(data.GQ_w(i) % h_it % XBZmu);
      }

      grad_alpha.col(i) = grad_alpha_tmp;

    }

    // penalty part //
    double fval = -1 * arma::accu(ELBO%data.samWt) + 0.5*ridge*arma::accu(alpha%alpha) +
      lam* arma::accu(gvec_sub % arma::abs(alpha));
    // g.col(0) = - 1* arma::sum(grad_alpha,1) + ridge*alpha +
    //   lam * gvec_sub % arma::sign(alpha);

    g.col(0) = - 1* grad_alpha * data.samWt + ridge*alpha +
      lam * gvec_sub % arma::sign(alpha);

    // double fval= -1*arma::accu(ELBO)/data.n;
    // g.col(0) = -1* arma::sum(grad_alpha,1)/data.n;

    return fval;
  }

};


// update alpha//
class PJFM_updateAlphaLasso_covBD_Fun{
public:
  const PJFM_data_t& data;
  const PJFM_para_covBD_t& para;
  arma::mat h_t;
  arma::mat XBZmu_T;
  arma::field<arma::vec> XBZmu_t;
  arma::field<arma::vec> ZVZ;
  int k_now = 0;

  double lam = 0;
  double ridge = 0;
  const arma::vec& gvec;

  PJFM_updateAlphaLasso_covBD_Fun(const PJFM_data_t& data,
                                  const PJFM_para_covBD_t& para,
                                  const arma::vec& gvec):
    data(data), para(para), gvec(gvec){

    h_t = arma::mat(data.GQ_t(0).n_elem,data.n, arma::fill::zeros);
    XBZmu_T = arma::mat(data.n, data.K, arma::fill::zeros);
    XBZmu_t = arma::field<arma::vec>(data.n, data.K);
    ZVZ = arma::field<arma::vec>(data.n, data.K);
    arma::vec tmp(data.GQ_t(0).n_elem, arma::fill::zeros);

    for(int i=0; i< data.n; i++){
      for(int k=0; k<data.K;k++){
        XBZmu_T(i,k) =  arma::accu(data.Z_T(i,k)%para.mu(i,k)) ;
        XBZmu_t(i,k) =  data.Z_t(i,k)*para.mu(i,k);
        for(int jj=0; jj<tmp.n_elem; jj++){
          tmp(jj) = arma::as_scalar(
            data.Z_t(i, k).row(jj) *  para.V(i,k)*
              data.Z_t(i, k).row(jj).t()
          );
        }
        ZVZ(i,k) = tmp;
      }
    }
  }

  void initiate(){

    int k=0;
    if(para.alpha_idx.n_elem>0){
      arma::vec tmp(data.GQ_t(0).n_elem, arma::fill::zeros);
      for(int i=0; i<data.n; i++){
        for(int j=0; j< para.alpha_idx.n_elem; j++){
          k = para.alpha_idx(j);
          XBZmu_T(i,k) = arma::accu(data.Z_T(i,k)%para.mu(i,k)) ;
          XBZmu_t(i,k) =  data.Z_t(i,k)*para.mu(i,k);
          for(int jj=0; jj<tmp.n_elem; jj++){
            tmp(jj) = arma::as_scalar(
              data.Z_t(i, k).row(jj) *  para.V(i,k)*
                data.Z_t(i, k).row(jj).t()
            );
          }
          ZVZ(i,k) = tmp;
        }
      }
    }


    for(int i=0; i< data.n; i++){

      arma::vec h_it = data.W_t(i) * para.beta0;

      for(int j=0; j< para.alpha_idx.n_elem; j++){
        k = para.alpha_idx(j);
        h_it += XBZmu_t(i,k) * para.alpha(k) +
          0.5 * para.alpha(k)*para.alpha(k) * ZVZ(i,k);
      }

      h_t.col(i) = h_it;
    }
  }

  void RemoveAdd(bool remove=true){

    for(int i=0; i< data.n; i++){
      arma::vec h_it = para.alpha(k_now)*XBZmu_t(i,k_now) +
        0.5 * para.alpha(k_now)*para.alpha(k_now) * ZVZ(i,k_now);

      if(remove){
        h_t.col(i) -= h_it;
      }else{
        h_t.col(i) += h_it;
      }
    }
  }

  // calculate the gradient for all k's
  // to determine the largest lasso lambda
  arma::vec gradKKT_all(){
    arma::vec grad(data.K,arma::fill::zeros);
    for(int k=0;k<data.K;k++){
      for(int i=0; i<data.n; i++){
        if(data.fstat(i)==1){
          grad(k) += data.samWt(i)*XBZmu_T(i,k);
        }
        arma::vec h_it = h_t.col(i);
        h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
        h_it = arma::exp(h_it);
        grad(k) -= data.samWt(i)*arma::accu(data.GQ_w(i) % h_it % XBZmu_t(i,k));
      }
    }
    return grad;
  }

  // calculate the gradient for k_now
  double gradKKT(){
    double grad=0;
    for(int i=0; i<data.n; i++){
      if(data.fstat(i)==1){
        grad += data.samWt(i)*XBZmu_T(i,k_now);
      }
      arma::vec h_it = h_t.col(i);
      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      grad -= data.samWt(i)*arma::accu(data.GQ_w(i) %h_it % XBZmu_t(i,k_now));
    }
    return grad;
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& para, arma::mat& g)
  {
    double alpha_k = para(0,0);

    double fval = 0;
    double grad = 0;
    for(int i=0; i<data.n; i++){
      if(data.fstat(i)==1){
        fval += data.samWt(i)*XBZmu_T(i,k_now)*alpha_k;
        grad += data.samWt(i)*XBZmu_T(i,k_now);
      }
      arma::vec h_it = h_t.col(i);
      h_it += alpha_k * XBZmu_t(i,k_now) +
        0.5*alpha_k*alpha_k*  ZVZ(i, k_now);
      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);

      arma::vec tmp = XBZmu_t(i,k_now) + alpha_k * ZVZ(i, k_now);

      fval -= data.samWt(i)*arma::accu(data.GQ_w(i) % h_it);
      grad -= data.samWt(i)*arma::accu(data.GQ_w(i) % h_it%tmp);
    }

    // penalty part
    if(alpha_k>0){
      fval -= lam*gvec(k_now)*alpha_k + 0.5*ridge*alpha_k*alpha_k;
      grad -= lam*gvec(k_now) + alpha_k*ridge;
    }else{
      fval -= -lam*gvec(k_now)*alpha_k + 0.5*ridge*alpha_k*alpha_k;
      grad -= -lam*gvec(k_now) + alpha_k*ridge;
    }

    fval = -fval;
    g(0,0) = -grad;

    return fval;
  }

};



// combine beta's into a vector //
// mu_i, V_i //
arma::vec PJFM_combineBeta_covBD(const PJFM_data_t& data,
                                 const PJFM_para_covBD_t& para){

  arma::field<arma::vec> beta_f(para.alpha_idx.n_elem);
  int k=0;
  for(int j=0; j< para.alpha_idx.n_elem;j++){
    k = para.alpha_idx(j);
    beta_f(j) = para.beta(k);
  }
  arma::vec beta_vec = field_to_vec(beta_f, para.p_x_vec_alpha);

  return arma::join_cols(para.beta0, beta_vec);
}

// combine beta's into a vector //
// mu_i, V_i //
void PJFM_storeBeta_covBD(arma::vec beta_all,
                          const PJFM_data_t& data,
                          PJFM_para_covBD_t& para){


  para.beta0 = beta_all.subvec(0, para.beta0.n_elem-1);
  arma::field<arma::vec> beta_f = vec_to_field(beta_all.subvec(para.beta0.n_elem, beta_all.n_elem-1), para.p_x_vec_alpha);

  int k=0;
  for(int j=0; j< para.alpha_idx.n_elem;j++){
    k = para.alpha_idx(j);
    para.beta(k) = beta_f(j);
  }

}

// update beta0 and beta's //
class PJFM_updateBetaALL_covBD_Fun{
public:
  const PJFM_data_t& data;
  const PJFM_para_covBD_t& para;

  PJFM_updateBetaALL_covBD_Fun(const PJFM_data_t& data,
                               const PJFM_para_covBD_t& para):
    data(data), para(para){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& beta_all, arma::mat& g)
  {

    arma::vec beta0 = beta_all(arma::span(0,para.beta0.n_elem-1), 0);
    arma::vec beta = beta_all(arma::span(para.beta0.n_elem , beta_all.n_rows-1), 0);
    arma::field<arma::vec> beta_f = vec_to_field(beta, para.p_x_vec_alpha);

    arma::mat grad_beta0(beta0.n_elem, data.n, arma::fill::zeros);
    arma::field<arma::vec> grad_beta(para.alpha_idx.n_elem);

    for(int k=0; k < para.alpha_idx.n_elem; k++){
      grad_beta(k) = arma::vec(beta_f(k).n_elem, arma::fill::zeros);
    }

    arma::vec ELBO(data.n, arma::fill::zeros);
    int k = 0;

    for(int i=0; i< data.n; i++){

      // recurrent
      for(int j=0; j< para.alpha_idx.n_elem; j++){

        k = para.alpha_idx(j);
        ELBO(i) += arma::accu(data.X(i,k) * beta_f(j));

        arma::vec h_it = data.X_t(i,k) * beta_f(j) +
          data.Z_t(i,k) * para.mu(i,k);
        for(int j=0; j< h_it.n_elem; j++){
          h_it(j) += 0.5 * arma::as_scalar(
            data.Z_t(i,k).row(j) *  para.V(i,k)*  data.Z_t(i,k).row(j).t()
          );
        }

        h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
        h_it = arma::exp(h_it);
        ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

        // gradient of beta//
        grad_beta(j) += data.samWt(i) * arma::sum(data.X(i,k).t(),1) -
          data.X_t(i,k).t() * (data.GQ_w(i) % h_it);
      }

      // surv
      if(data.fstat(i) == 1){
        ELBO(i) +=  arma::as_scalar(data.W_T.row(i) * beta0);
      }

      arma::vec h_it = data.W_t(i) * beta0;

      for(int j=0; j< para.alpha_idx.n_elem; j++){
        k = para.alpha_idx(j);
        h_it += data.Z_t(i,k)*para.mu(i,k)*para.alpha(k);
      }

      for(int jj=0; jj < h_it.n_elem; jj++){
        for(int j=0; j<para.alpha_idx.n_elem; j++){
          k = para.alpha_idx(j);
          h_it(jj) +=  0.5 * para.alpha(k)*para.alpha(k) * arma::as_scalar(
            data.Z_t(i, k).row(jj) *  para.V(i,k)*
              data.Z_t(i, k).row(jj).t()
          );
        }
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);

      ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

      // gradient of beta0//
      arma::vec grad_beta0_tmp(beta0.n_elem, arma::fill::zeros);
      if(data.fstat(i) == 1){
        grad_beta0_tmp += data.W_T.row(i).t();
      }
      grad_beta0_tmp -= data.W_t(i).t() * (data.GQ_w(i) % h_it);
      grad_beta0.col(i) = grad_beta0_tmp;

    }


    arma::vec grad_beta_vec = -1*field_to_vec(grad_beta, para.p_x_vec_alpha)/data.n;
    // arma::vec grad_beta0_vec =  -1*arma::sum(grad_beta0,1)/data.n;
    arma::vec grad_beta0_vec =  -1 * grad_beta0 * data.samWt /data.n;
    double fval= -1*arma::accu(ELBO % data.samWt)/data.n;
    g.col(0) = arma::join_cols(grad_beta0_vec, grad_beta_vec);

    return fval;
  }

};

// update beta0 //
class PJFM_updateBeta0_covBD_Fun{
public:
  const PJFM_data_t& data;
  const PJFM_para_covBD_t& para;

  PJFM_updateBeta0_covBD_Fun(const PJFM_data_t& data,
                             const PJFM_para_covBD_t& para):
    data(data), para(para){
  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& beta_all, arma::mat& g)
  {

    arma::vec beta0 = beta_all.col(0);
    arma::mat grad_beta0(beta0.n_elem, data.n, arma::fill::zeros);
    arma::vec ELBO(data.n, arma::fill::zeros);

    for(int i=0; i< data.n; i++){

      // surv
      if(data.fstat(i) == 1){
        ELBO(i) +=  arma::as_scalar(data.W_T.row(i) * beta0);
      }

      arma::vec h_it = data.W_t(i) * beta0;
      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);

      ELBO(i) -=  arma::accu(data.GQ_w(i) % h_it);

      // gradient of beta0//
      arma::vec grad_beta0_tmp(beta0.n_elem, arma::fill::zeros);
      if(data.fstat(i) == 1){
        grad_beta0_tmp += data.W_T.row(i).t();
      }
      grad_beta0_tmp -= data.W_t(i).t() * (data.GQ_w(i) % h_it);
      grad_beta0.col(i) = grad_beta0_tmp;

    }

    //arma::vec grad_beta0_vec =  -1*arma::sum(grad_beta0,1)/data.n;
    arma::vec grad_beta0_vec =  -1* grad_beta0*data.samWt /data.n;
    double fval= -1*arma::accu(ELBO % data.samWt)/data.n;
    g.col(0) = grad_beta0_vec;

    return fval;
  }

};


// combine parameters into vector //
// mu_i, V_i //
arma::vec PJFM_combineMuV_covBD(const PJFM_data_t& data,
                                const PJFM_para_covBD_t& para,
                                const int& i){

  arma::field<arma::vec> mu_f(para.alpha_idx.n_elem);
  arma::field<arma::vec> Lvec_f(para.alpha_idx.n_elem);

  int k;
  for(int j=0; j<para.alpha_idx.n_elem; j++){
    k = para.alpha_idx(j);
    mu_f(j) = para.mu(i,k);
    Lvec_f(j) = para.Lvec(i,k);
  }

  arma::vec mu = field_to_vec(mu_f, para.p_z_vec_alpha);
  arma::vec Lvec =field_to_vec(Lvec_f, para.p_zz_vec_alpha);

  arma::vec muV(Lvec.n_elem + mu.n_elem);
  muV.subvec(0, mu.n_elem - 1) = mu;
  muV.subvec(mu.n_elem, muV.n_elem-1) = Lvec;
  return muV;
}

// to put the new updates into para //
void PJFM_storeMuV_covBD(const PJFM_data_t& data,
                         PJFM_para_covBD_t& para,
                         const arma::vec& muV, const int& i){

  arma::vec mu =  muV.subvec(0, para.p_z_alpha - 1);
  arma::vec Lvec = muV.subvec(mu.n_elem, muV.n_elem-1);

  arma::field<arma::vec> mu_f = vec_to_field(mu, para.p_z_vec_alpha);
  arma::field<arma::vec> Lvec_f = vec_to_field_L(Lvec, para.p_z_vec_alpha);

  int k=0;
  for(int j=0; j< para.alpha_idx.n_elem;j++){
    k = para.alpha_idx(j);
    para.mu(i,k) = mu_f(j);
    para.Lvec(i,k) = Lvec_f(j);
    arma::mat L = makeLowTriMat(para.V(i,k),  Lvec_f(j));
    para.V(i,k) = L*L.t();
  }
}

// update variational parameters mu_i and V_i //
// only update mu_ik and V_ik for nonzero alpha_k //
class PJFM_updateMuV_covBD_Fun{
public:
  const PJFM_data_t& data;
  const PJFM_para_covBD_t& para;

  arma::vec Z_ia_T;
  arma::mat Z_ia_t;

  int i = 0;

  PJFM_updateMuV_covBD_Fun(const PJFM_data_t& data,
                           const PJFM_para_covBD_t& para) :
    data(data), para(para){
  }

  void updateIntermediate(){

    Z_ia_T = field_to_alpha_vec(data.Z_T, para.alpha,
                                i, data.p_z_vec,para.alpha_idx);
    Z_ia_t = field_to_alpha_mat(data.Z_t, para.alpha,
                                i, data.p_z_vec,para.alpha_idx);

  }

  // Return the objective function with gradient.
  double EvaluateWithGradient(const arma::mat& muV, arma::mat& g)
  {
    arma::vec mu = muV(arma::span(0,para.p_z_alpha-1), 0);
    arma::vec Lvec = muV(arma::span(para.p_z_alpha, muV.n_rows-1), 0);

    arma::field<arma::vec> mu_f = vec_to_field(mu,  para.p_z_vec_alpha);
    arma::field<arma::vec> Lvec_f = vec_to_field_L(Lvec, para.p_z_vec_alpha);

    int k=0;
    arma::field<arma::mat> V_f(para.alpha_idx.n_elem);
    arma::field<arma::mat> L_f(para.alpha_idx.n_elem);

    for(int j=0; j< para.alpha_idx.n_elem;j++){
      k = para.alpha_idx(j);
      L_f(j) = makeLowTriMat(para.V(i,k),  Lvec_f(j));
      V_f(j) = L_f(j)*L_f(j).t();
    }

    double val;
    double sign;


    /// fun value
    double fval = 0.0;
    arma::vec grad_mu(mu.n_rows, arma::fill::zeros);
    // arma::mat grad_V(L.n_rows, L.n_rows, arma::fill::zeros);
    // arma::mat V_tmp = V;
    // V_tmp.zeros();
    arma::field<arma::mat> grad_V(para.alpha_idx.n_elem);

    // recurrent
    //Rcout << "recurrent\n";
    int start =0;

    for(int j =0; j<para.alpha_idx.n_elem;j++){

      k = para.alpha_idx(j);
      //Rcout << "j="<< j << "; k=" << k << "\n";


      fval += arma::accu(data.Z(i,k) * mu_f(j));

      arma::vec h_it = data.X_t(i,k) * para.beta(k) +
        data.Z_t(i,k) * mu_f(j);
      for(int jj=0; jj< h_it.n_elem; jj++){
        h_it(jj) += 0.5 * arma::as_scalar(
          data.Z_t(i,k).row(jj) *
            V_f(j)* data.Z_t(i,k).row(jj).t()
        );
      }

      h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
      h_it = arma::exp(h_it);
      fval -=  arma::accu(data.GQ_w(i) % h_it);

      //Rcout << "grd\n";

      grad_mu.subvec(start, start+para.p_z_vec_alpha(j)-1) =
        arma::sum(data.Z(i,k).t(), 1) - data.Z_t(i,k).t() * (data.GQ_w(i) % h_it);
      start = start + para.p_z_vec_alpha(j);

      grad_V(j) = - data.Z_t(i,k).t() *
        arma::diagmat((data.GQ_w(i) % h_it)) *
        data.Z_t(i,k) * L_f(j);
    }


    // surv
    //Rcout << "surv\n";
    if(data.fstat(i) == 1){

      for(int j =0; j<para.alpha_idx.n_elem;j++){
        k = para.alpha_idx(j);
        fval +=  arma::accu(data.Z_T(i,k)%mu_f(j)) *para.alpha(k);
      }

      grad_mu += Z_ia_T;
    }

    arma::vec h_it = data.W_t(i) * para.beta0 + Z_ia_t*mu;;
    for(int jj=0;jj<h_it.n_elem; jj++){
      for(int j=0; j<para.alpha_idx.n_elem;j++){
        k = para.alpha_idx(j);
        h_it(jj) +=  0.5 * para.alpha(k)*para.alpha(k) * arma::as_scalar(
          data.Z_t(i, k).row(jj) *  V_f(j)*
            data.Z_t(i, k).row(jj).t()
        );
      }
    }

    h_it = arma::clamp(h_it, -MAX_EXP, MAX_EXP);
    h_it = arma::exp(h_it);

    fval -=  arma::accu(data.GQ_w(i) % h_it);

    grad_mu -=  Z_ia_t.t() * (data.GQ_w(i) % h_it);

    for(int j=0; j<para.alpha_idx.n_elem;j++){
      k = para.alpha_idx(j);
      grad_V(j) -=
        para.alpha(k)*para.alpha(k) *  data.Z_t(i, k).t() *
        arma::diagmat((data.GQ_w(i) % h_it)) *
        data.Z_t(i, k) * L_f(j) ;
    }

    // variational part
    //Rcout << "variational\n";
    for(int j=0; j<para.alpha_idx.n_elem;j++){
      k = para.alpha_idx(j);
      arma::log_det(val, sign, V_f(j));
      fval += -0.5 *  arma::as_scalar(mu_f(j).t() * para.invSigma(k) *mu_f(j))
        -0.5*arma::trace(para.invSigma(k)*V_f(j)) +  0.5 * val;
    }

    start = 0;
    for(int j=0; j<para.alpha_idx.n_elem; j++){
      k = para.alpha_idx(j);
      grad_mu.subvec(start, start+para.p_z_vec_alpha(j)-1) -=
        para.invSigma(k) * mu_f(j);
      start = start + para.p_z_vec_alpha(j);

      grad_V(j) += arma::trans(arma::inv( arma::trimatl(L_f(j)))) -
        para.invSigma(k)*L_f(j);
    }

    /// gradient

    //Rcout << "gradient\n";
    fval = -1*fval;

    g(arma::span(0,para.p_z_alpha-1), 0) = -grad_mu;
    g(arma::span(para.p_z_alpha, muV.n_rows-1), 0) = -LowTriVec_field(grad_V);


    //Rcout << fval << "\n";

    return fval;
  }

};


// update  Sigma //
void PJFM_updateSig_covBD(const PJFM_data_t& data,
                          PJFM_para_covBD_t& para){

  int k =0;
  if(para.alpha_idx.n_elem > 0){
    for(int j=0; j< para.alpha_idx.n_elem;j++){
      k = para.alpha_idx(j);
      para.Sigma(k).zeros();
      for(int i=0; i< data.n; i++){
        //para.Sigma(k) += para.mu(i,k) * para.mu(i,k).t() + para.V(i,k);
        para.Sigma(k) += data.samWt(i) * (para.mu(i,k) * para.mu(i,k).t() + para.V(i,k));
      }
      // para.Sigma(k) /= data.n;
      para.Sigma(k) /= arma::accu( data.samWt);
      para.invSigma(k) = myinvCpp(para.Sigma(k));
    }
  }
}


// combine all parameters into a vector
arma::vec PJFM_combinaPara_covBD(const PJFM_data_t& data,
                                 const PJFM_para_covBD_t& para){

  arma::vec beta_all = arma::join_cols(para.beta0, field_to_vec(para.beta, data.p_x_vec));
  arma::vec sig_vec = LowTriVec_field(para.Sigma);
  return arma::join_cols(para.alpha, beta_all, sig_vec);

}

bool check_para(const PJFM_para_covBD_t& para){
  bool na_yes = false;

  if(para.beta0.has_nan()){
    Rcout << "beta0 has nan \n";
    na_yes = true;
  }

  if(para.alpha.has_nan()){
    Rcout << "alpha has nan \n";
    na_yes = true;
  }

  for(int k=0; k<para.Sigma.n_elem; k++){
    if(para.Sigma(k).has_nan()){
      Rcout << "Simga has nan \n";
      na_yes = true;
    }
  }

  for(int k=0; k<para.beta.n_elem; k++){
    if(para.beta(k).has_nan()){
      Rcout << "Beta has nan \n";
      na_yes = true;
    }
  }

  for(int i=0; i<para.mu.n_rows; i++ ){
    for(int k=0; k <para.mu.n_cols; k++ ){
      if(para.mu(i,k).has_nan()){
        Rcout << "Mu has nan \n";
        na_yes = true;
      }
    }
  }


  for(int i=0; i<para.V.n_rows; i++ ){
    for(int k=0; k <para.V.n_cols; k++ ){
      if(para.V(i,k).has_nan()){
        Rcout << "V has nan \n";
        na_yes = true;
      }
    }
  }


  return na_yes;

}


//' Main function to run PJFM given one lasso penalty
//' @noRd
//'
// [[Rcpp::export]]
List PJFM_covBD(const List& datalist, const List& paralist,
                const arma::vec& gvec, double lam, double ridge,
                int maxiter = 100, double eps=1e-4){

  //Rcout << "1\n";
  PJFM_data_t data(datalist);
  PJFM_para_covBD_t para(paralist);
  para.NonZeroAlpha();

  ens::L_BFGS lbfgs;
  lbfgs.MinGradientNorm() = MinGradientNorm;
  lbfgs.Factr() = Factr;

  PJFM_updateMuV_covBD_Fun MuV_fun(data, para);
  PJFM_updateBetaALL_covBD_Fun betaAll_fun(data,  para);
  PJFM_updateAlphaLasso_covBD_Fun Alpha_fun(data,  para, gvec);
  Alpha_fun.lam = lam;
  Alpha_fun.ridge = ridge;
  Alpha_fun.initiate();

  // arma::vec grad_all  = Alpha_fun.gradKKT_all();
  // double lam_max = (grad_all/gvec).max();
  // Rcout << "lam_max: " << lam_max << "\n";

  arma::mat ELBO_base = PJFM_calcELBO_baseRecur(data, para);
  double ELBO = PJFM_calcELBO_covBD(data, para, ELBO_base, gvec, lam, ridge);

  arma::vec ELBO_vec(maxiter);
  int iter;
  //Rcout << ELBO << "\n";

  double grd = 0;
  arma::mat alpha_k(1,1);
  arma::vec alpha_vec = para.alpha;

  arma::vec para_pre = PJFM_combinaPara_covBD(data, para);
  arma::vec para_aft = PJFM_combinaPara_covBD(data, para);

  for(iter=0;iter<maxiter;iter++){

    para_pre = PJFM_combinaPara_covBD(data, para);

    // update alpha's
    // Rcout << "iteration:" << iter << "...................\n";

    // Rcout << "update alpha \n";
    Alpha_fun.initiate();
    for(int j = 0;j<maxiter; j++){
      alpha_vec = para.alpha;
      for(int k=0; k<data.K; k++){
        Alpha_fun.k_now = k;
        if(para.alpha(k) !=0){
          Alpha_fun.RemoveAdd(true);
        }
        grd = Alpha_fun.gradKKT();
        if(std::fabs(grd) < lam*gvec(k)){
          alpha_k(0,0) = 0;
        }else{
          alpha_k(0,0) = para.alpha(k);
          lbfgs.Optimize(Alpha_fun,alpha_k);
        }
        para.alpha(k) = alpha_k(0,0);

        if(para.alpha(k) !=0){
          Alpha_fun.RemoveAdd(false);
        }
      }
      double err_alpha = std::sqrt(arma::accu(arma::square(alpha_vec-para.alpha)));
      if(err_alpha<eps){
        break;
      }
    }
    para.NonZeroAlpha();

    // ELBO = PJFM_calcELBO_covBD(data, para, ELBO_base, gvec, lam, ridge);
    // Rcout << ELBO << "\n";
    // Rcout << para.alpha.t() << "\n";
    // Rcout << para.alpha_idx.t() << "\n";

    // update other parameters
    if(para.alpha_idx.n_elem>0){
      // update beta's
      // Rcout << "update beta\n";
      arma::vec beta_all = PJFM_combineBeta_covBD(data,para);
      lbfgs.Optimize(betaAll_fun, beta_all);

      PJFM_storeBeta_covBD(beta_all, data,  para);

      // ELBO = PJFM_calcELBO_covBD(data, para, ELBO_base, gvec, lam, ridge);
      // Rcout << ELBO << "\n";

      // update Sigma
      //Rcout << "update sigma\n";
      PJFM_updateSig_covBD(data,  para);


      //Rcout << "PJFM_calcELBO_covBD\n";
      // ELBO = PJFM_calcELBO_covBD(data, para, ELBO_base, gvec, lam, ridge);
      // Rcout << ELBO << "\n";

      // update V and mu -- variational para
      //Rcout << "update V and mu\n";

      for(int i=0; i < data.n; i++){
        MuV_fun.i = i;
        MuV_fun.updateIntermediate();
        arma::vec muV = PJFM_combineMuV_covBD(data, para, i);
        lbfgs.Optimize(MuV_fun,muV);
        PJFM_storeMuV_covBD(data, para,  muV, i);
      }

    }

    // calculate errors
    para_aft = PJFM_combinaPara_covBD(data, para);

    double err_para = std::sqrt(arma::accu(arma::square(para_aft-
                                para_pre))/para_pre.n_elem);
    if(err_para < eps){
      //Rcout << "error: " << err_para << "\n";
      break;
    }
  }

  ELBO = PJFM_calcELBO_covBD(data, para, ELBO_base, gvec, lam, ridge);
  //Rcout <<"#iter: "<< iter << "; ELBO: " << ELBO << "\n";

  return List::create(
    _["Sigma"] = para.Sigma,
    _["alpha"] = para.alpha,
    _["beta"] = para.beta,
    _["beta0"] = para.beta0,
    _["mu"] = para.mu,
    _["V"] = para.V,
    _["ELBO_vec"] = ELBO_vec,
    _["iter"] = iter,
    _["ELBO"] = ELBO
  );
}

double PJFM_covBD_base(const PJFM_data_t& data, PJFM_para_covBD_t& para,
                       const arma::mat& ELBO_base,
                       const arma::vec& gvec, double lam, double ridge,
                       int maxiter = 100, double eps=1e-4){
  para.NonZeroAlpha();

  ens::L_BFGS lbfgs;
  lbfgs.MinGradientNorm() = MinGradientNorm;
  lbfgs.Factr() = Factr;

  PJFM_updateMuV_covBD_Fun MuV_fun(data, para);
  PJFM_updateBetaALL_covBD_Fun betaAll_fun(data,  para);
  PJFM_updateAlphaLasso_covBD_Fun Alpha_fun(data,  para, gvec);
  Alpha_fun.lam = lam;
  Alpha_fun.ridge = ridge;
  Alpha_fun.initiate();

  int iter;
  double grd = 0;
  arma::mat alpha_k(1,1);
  arma::vec alpha_vec = para.alpha;

  arma::vec para_pre = PJFM_combinaPara_covBD(data, para);
  arma::vec para_aft = PJFM_combinaPara_covBD(data, para);

  for(iter=0;iter<maxiter;iter++){

    para_pre = PJFM_combinaPara_covBD(data, para);

    // update alpha's
    //Rcout << "1\n";
    Alpha_fun.initiate();
    for(int j = 0;j<maxiter; j++){
      alpha_vec = para.alpha;
      for(int k=0; k<data.K; k++){
        Alpha_fun.k_now = k;
        if(para.alpha(k) !=0){
          Alpha_fun.RemoveAdd(true);
        }
        grd = Alpha_fun.gradKKT();
        if(std::fabs(grd) < lam*gvec(k)){
          alpha_k(0,0) = 0;
        }else{
          alpha_k(0,0) = para.alpha(k);
          lbfgs.Optimize(Alpha_fun,alpha_k);
        }
        para.alpha(k) = alpha_k(0,0);

        if(para.alpha(k) !=0){
          Alpha_fun.RemoveAdd(false);
        }
      }
      double err_alpha = std::sqrt(arma::accu(arma::square(alpha_vec-para.alpha)));
      if(err_alpha<eps){
        break;
      }
    }
    para.NonZeroAlpha();

    // update other parameters
    if(para.alpha_idx.n_elem>0){

      //Rcout << "2\n";
      // update beta's
      arma::vec beta_all = PJFM_combineBeta_covBD(data,para);
      lbfgs.Optimize(betaAll_fun, beta_all);
      PJFM_storeBeta_covBD(beta_all, data,  para);

      //Rcout << "3\n";
      // update Sigma
      PJFM_updateSig_covBD(data,  para);

      //Rcout << "4\n";
      // update V and mu -- variational para

      for(int i=0; i < data.n; i++){

        MuV_fun.i = i;
        MuV_fun.updateIntermediate();
        arma::vec muV = PJFM_combineMuV_covBD(data, para, i);
        lbfgs.Optimize(MuV_fun,muV);
        PJFM_storeMuV_covBD(data, para,  muV, i);
      }
    }



    // calculate errors
    para_aft = PJFM_combinaPara_covBD(data, para);

    double err_para = std::sqrt(arma::accu(arma::square(para_aft-
                                para_pre))/para_pre.n_elem);
    if(err_para < eps){
      break;
    }
  }

  double ELBO = PJFM_calcELBO_covBD(data, para, ELBO_base, gvec, 0, 0);

  //Rcout <<"#iter = "<<iter << "; ELBO = "<< ELBO << "\n";

  return ELBO;

}


void PJFM_covBD_init(const PJFM_data_t& data, PJFM_para_covBD_t& para){

  para.NonZeroAlpha();
  ens::L_BFGS lbfgs;
  lbfgs.MinGradientNorm() = MinGradientNorm;
  lbfgs.Factr() = Factr;

  PJFM_updateBeta0_covBD_Fun betaAll_fun(data,  para);

  arma::vec beta0 = para.beta0;
  lbfgs.Optimize(betaAll_fun, beta0);
  para.beta0 = beta0;
}



//' Main function to run PJFM for a sequence of lasso penalties
//' @noRd
//'
// [[Rcpp::export]]
List PJFM_covBD_seq(const List& datalist, const List& paralist,
                    const arma::vec& gvec, int nlam,
                    double ridge, int pmax,
                    const double min_ratio=0.001, const int maxiter=100,
                    const double eps=1e-4, const bool UseSurvN=true){

  PJFM_data_t data(datalist);
  PJFM_para_covBD_t para(paralist);
  para.NonZeroAlpha();

  double lam_max = PJFM_get_lammax_covBD(data, para, gvec);
  //Rcout << "lam_max: " << lam_max << "\n";

  //Rcout << "initiation ... \n";
  PJFM_covBD_init(data, para);
  ///Rcout << "re-calculate lam_max ... \n";
  lam_max = PJFM_get_lammax_covBD(data, para, gvec);
  //Rcout << "lam_max after initiation: " << lam_max << "\n";

  arma::vec lam_seq = arma::exp(arma::linspace(
    std::log(lam_max), std::log(lam_max*min_ratio), nlam ));

  arma::mat ELBO_base = PJFM_calcELBO_baseRecur(data, para);

  // object to store the results
  arma::mat alpha_mat(data.K, nlam, arma::fill::zeros);
  arma::vec ELBO_vec(nlam, arma::fill::zeros);
  arma::vec BIC(nlam, arma::fill::zeros);

  arma::field<arma::vec> mu_f(para.mu.n_rows, para.mu.n_cols,nlam); // n \times K vec
  arma::field<arma::mat> V_f(para.V.n_rows, para.V.n_cols, nlam); // n \times K mat
  arma::field<arma::vec> beta_f(para.beta.n_rows, nlam); // K \times 1 vec
  arma::field<arma::mat> Sigma_f(para.Sigma.n_rows, nlam); // K \times 1 mat
  arma::mat beta0_mat(para.beta0.n_elem, nlam); // dim = p_w

  // run the algorithm
  int j=0;
  double eff_sam;
  if(UseSurvN){
    eff_sam = arma::accu(data.fstat);
  }else{
    eff_sam = data.ftime.n_elem*1.0;
  }

  for(j=0; j<nlam; j++){
    double lam = lam_seq(j);
    //Rcout << j+1 << "th lam=" << lam <<"\n";
    ELBO_vec(j) = PJFM_covBD_base(data,  para, ELBO_base, gvec,
             lam,  ridge, maxiter, eps);

    //Rcout << "write results \n";

    alpha_mat.col(j) = para.alpha;
    beta0_mat.col(j) = para.beta0;
    mu_f.slice(j) = para.mu;
    V_f.slice(j) = para.V;
    beta_f.col(j) = para.beta;
    Sigma_f.col(j) = para.Sigma;

    BIC(j) = -2*ELBO_vec(j) + std::log(eff_sam) * para.alpha_idx.n_elem;
    // Rcout << para.alpha.t() << "\n";
    // Rcout << para.alpha_idx.t() << "\n";

    if(para.alpha_idx.n_elem > pmax){
      break;
    }

    // bool na_yes = check_para(para);
    //
    // if(na_yes){
    //   Rcout << "nan found in the para\n";
    //   break;
    // }else{
    //   Rcout << "NO nan found in the para\n";
    // }

  }

  // Rcout << "shed_cols \n" ;
  if(j+1 < nlam){
    alpha_mat.shed_cols(j+1, nlam-1);
    //gamma_mat.shed_cols(j+1, nlam-1);
    //weib_mat.shed_cols(j+1, nlam-1);
    //sig2_mat.shed_cols(j+1, nlam-1);
    ELBO_vec.shed_rows(j+1, nlam-1);
    BIC.shed_rows(j+1, nlam-1);
    lam_seq.shed_rows(j+1, nlam-1);
  }

  arma::uword ind_min = BIC.index_min();

  // Rcout << "ind_min \n" ;

  para.mu = mu_f.slice(ind_min);
  para.V = V_f.slice(ind_min);
  para.Sigma = Sigma_f.col(ind_min);
  para.alpha = alpha_mat.col(ind_min);
  para.beta =  beta_f.col(ind_min);
  para.beta0 = beta0_mat.col(ind_min);

  return List::create(
    _["alpha_mat"] = alpha_mat,
    _["ELBO"] = ELBO_vec,
    _["BIC"] = BIC,
    _["lam_seq"] = lam_seq,
    _["Sigma"] = para.Sigma,
    _["alpha"] = para.alpha,
    _["beta"] = para.beta,
    _["beta0"] = para.beta0
    //_["mu"] = para.mu,
    //_["V"] = para.V
  );

}




