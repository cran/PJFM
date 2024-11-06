## PFJM wrapper functions ##

#' Simulated Recurrent Events Data
#'
#' This dataset contains recurrent events data.
#'
#' \itemize{
#'   \item ID: patient ID
#'   \item feature_id: types of recurrent events
#'   \item time: occurrence time
#' }
#'
#' @name RecurData
#' @docType data
#' @author Jiehuan Sun \email{jiehuan.sun@gmail.com}
#' @keywords data
#' @usage data(PJFMdata)
#' @format A data frame with 57582 rows and 3 variables
NULL

#' Simulated Survival Data
#'
#' This dataset contains survival outcome.
#'
#' \itemize{
#'   \item ID: patient ID
#'   \item fstat: censoring indicator
#'   \item ftime: survival time
#'   \item x: baseline covariates
#' }
#' @name SurvData
#' @docType data
#' @author Jiehuan Sun \email{jiehuan.sun@gmail.com}
#' @keywords data
#' @usage data(PJFMdata)
#' @format A data frame with 300 rows and 4 variables
NULL


#' control_list
#'
#' This list contains a list of parameters specifying the joint frailty model.
#'
#' \itemize{
#'   \item ID_name: the variable name indicating the patient ID in both
#'   recurrent events data and survival data.
#'   \item item_name: the variable name indicating the types of recurrent events
#'   in the recurrent events data.
#'   \item time_name: the variable name indicating the occurrence time in the
#'   recurrent events data.
#'   \item fix_cov: a set of variables names indicating the covariates of
#'   fixed-effects in the recurrent events submodel.
#'   If NULL, not baseline covariates are included.
#'   \item random_cov: a set of variables names indicating the covariates of
#'   random-effects in the recurrent events submodel.
#'   If NULL, not baseline covariates are included.
#'   \item recur_fix_time_fun: a function specifying the time-related basis functions (fixed-effects) in
#'   the recurrent events submodel.
#'   \item recur_ran_time_fun: a function specifying the time-related basis functions (random-effects) in
#'   the recurrent events submodel. If this is an intercept only function,
#'   then only a random intercept is included (i.e. a joint frailty model).
#'   \item surv_fix_time_fun: a log-hazard function for the survival submodel.
#'   \item surv_time_name the variable name for the survival time
#'   in the survival data.
#'   \item surv_status_name the variable name for the censoring indicator
#'   in the survival data.
#'   \item surv_cov a set of variables names specifying the baseline covariates
#'   in the survival submodel.
#'   \item n_points an integer indicating the numebr of nodes being used in
#'   the Gaussian quadrature.
#' }
#' @name control_list
#' @docType data
#' @author Jiehuan Sun \email{jiehuan.sun@gmail.com}
#' @keywords data
NULL


#' @noRd
#' @keywords internal
prep_data <- function(RecurData=NULL, SurvData = NULL,
                      control_list=NULL, marker.name=NULL){

    ### control_list
    ID_name = control_list$ID_name
    item_name = control_list$item_name
    ##value_name = control_list$value_name
    time_name = control_list$time_name
    fix_cov = control_list$fix_cov
    random_cov = control_list$random_cov
    recur_fix_time_fun = control_list$recur_fix_time_fun
    recur_ran_time_fun = control_list$recur_ran_time_fun
    surv_fix_time_fun = control_list$surv_fix_time_fun
    surv_time_name = control_list$surv_time_name
    surv_status_name = control_list$surv_status_name
    surv_cov =  control_list$surv_cov
    n_points =  control_list$n_points

    ###
    data.list = list()
    para.list = list()

    fix_est_name = c(colnames(recur_fix_time_fun(1)), fix_cov)
    rand_est_name = c(colnames(recur_ran_time_fun(1)), random_cov)
    surv_est_name = c(colnames(surv_fix_time_fun(1)), surv_cov)

    ## process recurrent event data
    ## N_ij

    uni_ID = SurvData[,ID_name]
    if(is.null(marker.name)){
        marker.name = sort(unique(RecurData[,item_name]))
    }
    RecurData = RecurData[is.element(RecurData[,item_name],marker.name),]
    N_mat = sparseMatrix( i = match(RecurData[,ID_name], uni_ID),
                          j = match(RecurData[,item_name], marker.name), x= 1,
                          dims = c(length(uni_ID), length(marker.name)))
    N_mat = as.matrix(N_mat)

    ## X_i, Z_i
    X_recur_list = lapply(uni_ID, function(i){

        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        lapply(marker.name,function(x){

            time.tmp = data.tmp[data.tmp[,item_name]==x,time_name]
            if(length(time.tmp) !=0){
                time_mat.tmp = recur_fix_time_fun(time.tmp)
            }else{
                time_mat.tmp = recur_fix_time_fun(0)
                time_mat.tmp[,] = 0
            }
            if(!is.null(fix_cov)){
                cov.tmp = SurvData[SurvData[,ID_name]==i, fix_cov, drop=FALSE]
                cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])

                mat = as.matrix(cbind( time_mat.tmp,cov.tmp ))
                if(length(time.tmp) ==0){
                    mat[,] = 0
                }
                mat

            }else{
                as.matrix(cbind( time_mat.tmp ))
            }
            #cbind(1,data.tmp$years[data.tmp$item==x])
        })
    })

    X_recur_list = do.call(rbind, X_recur_list)


    Z_recur_list = lapply(uni_ID, function(i){

        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        lapply(marker.name,function(x){

            time.tmp = data.tmp[data.tmp[,item_name]==x,time_name]
            if(length(time.tmp) !=0){
                time_mat.tmp = recur_ran_time_fun(time.tmp)
            }else{
                time_mat.tmp = recur_ran_time_fun(0)
                time_mat.tmp[,] = 0
            }
            if(!is.null(random_cov)){
                cov.tmp = SurvData[SurvData[,ID_name]==i, random_cov, drop=FALSE]
                cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
                mat = as.matrix(cbind( time_mat.tmp,cov.tmp ))

                if(length(time.tmp) ==0){
                    mat[,] = 0
                }
                mat

            }else{
                as.matrix(cbind( time_mat.tmp ))
            }
            #cbind(1,data.tmp$years[data.tmp$item==x])
        })
    })

    Z_recur_list = do.call(rbind, Z_recur_list)

    ##   Z_i(T_i)
    Z_T_recur_list = lapply(uni_ID, function(i){
        T_i = SurvData[SurvData[,ID_name] == i,surv_time_name]
        #T_i = SurvData$ftime[SurvData$ID==i]
        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        lapply(marker.name,function(x){
            vv = SurvData[SurvData[,ID_name]==i, random_cov, drop=FALSE]
            matrix(c(as.numeric(recur_ran_time_fun(T_i)), as.numeric(vv[1,]) ), ncol=1)
            #matrix(c(1,T_i),ncol=1)
        })
    })
    Z_T_recur_list = do.call(rbind, Z_T_recur_list)

    ## X_i(t) , z_i(t)
    ## this depends on the number of legendre Gaussian quadrature points
    Gauss.point  = gauss.quad(n_points)
    # \int_0^{T_i} f(t)dt
    # t_node = Gauss.point$nodes *(Ti/2) + Ti/2
    # w_node = Gauss.point$weights
    # Ti/2 * sum(w_node * f(t_node))

    X_t_recur_list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        t_node = Gauss.point$nodes *(Ti/2) + Ti/2
        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        time_mat.tmp = recur_fix_time_fun(t_node)

        lapply(marker.name,function(x){
            if(!is.null(fix_cov)){
                cov.tmp = SurvData[SurvData[,ID_name]==i, fix_cov, drop=FALSE]
                cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
                as.matrix(cbind(time_mat.tmp, cov.tmp ))
            }else{
                as.matrix(cbind( time_mat.tmp ))
            }
        })
    })

    X_t_recur_list = do.call(rbind, X_t_recur_list)

    Z_t_recur_list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        t_node = Gauss.point$nodes *(Ti/2) + Ti/2
        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        time_mat.tmp = recur_ran_time_fun(t_node)

        lapply(marker.name,function(x){

            if(!is.null(random_cov)){
                cov.tmp = SurvData[SurvData[,ID_name]==i, random_cov, drop=FALSE]
                cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
                as.matrix(cbind(  time_mat.tmp[,,drop=FALSE], cov.tmp ))
            }else{
                as.matrix(cbind(time_mat.tmp[,,drop=FALSE] ))
            }
        })
    })

    Z_t_recur_list = do.call(rbind, Z_t_recur_list)


    w_node.list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        Gauss.point$weights*Ti/2
    })
    t_node.list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        Gauss.point$nodes *(Ti/2) + Ti/2
    })


    ## survival data

    ##   X_i(T_i)
    X_T_surv_list = lapply(uni_ID, function(i){
        T_i = SurvData[SurvData[,ID_name] == i,surv_time_name]
        #T_i = SurvData$ftime[SurvData$ID==i]
        #data.tmp =  RecurData[RecurData[,ID_name]==i,]
        vv = SurvData[SurvData[,ID_name]==i, surv_cov, drop=FALSE]

        c(as.numeric(surv_fix_time_fun(T_i)), as.numeric(vv[1,]) )
        #matrix(c(1,T_i),ncol=1)

    })
    X_T_surv_list = do.call(rbind, X_T_surv_list)

    ## X_i(t)
    X_t_surv_list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        t_node = Gauss.point$nodes *(Ti/2) + Ti/2
        #data.tmp =  RecurData[RecurData[,ID_name]==i,]
        time_mat.tmp = surv_fix_time_fun(t_node)

        if(!is.null(surv_cov)){
            cov.tmp = SurvData[SurvData[,ID_name]==i, surv_cov, drop=FALSE]
            cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
            list(as.matrix(cbind(time_mat.tmp, cov.tmp )))
        }else{
            list(as.matrix(cbind( time_mat.tmp )))
        }

    })

    X_t_surv_list = do.call(rbind, X_t_surv_list)

    samWt = rep(1,length(uni_ID))
    ## output list

    data.list[["N"]] = N_mat

    data.list[["X"]] = X_recur_list
    data.list[["Z"]] = Z_recur_list
    data.list[["Z_T"]] = Z_T_recur_list
    data.list[["X_t"]] = X_t_recur_list
    data.list[["Z_t"]] = Z_t_recur_list

    data.list[["W_T"]] = X_T_surv_list
    data.list[["W_t"]] = X_t_surv_list

    data.list[["GQ_w"]] = w_node.list
    data.list[["GQ_t"]] = t_node.list
    data.list[["ftime"]] = SurvData[,surv_time_name]
    data.list[["fstat"]] = SurvData[,surv_status_name]
    data.list[["samWt"]] = samWt

    list(data.list=data.list, uni_ID=uni_ID, marker.name=marker.name,
         fix_est_name=fix_est_name,rand_est_name=rand_est_name,
         surv_est_name=surv_est_name)

}

#' @noRd
#' @keywords internal
PJFM_init <- function(RecurData=NULL, SurvData = NULL,
                      control_list=NULL, marker.name=NULL){

    data.list = prep_data(RecurData=as.data.frame(RecurData),
                          SurvData = as.data.frame(SurvData),
                          control_list=control_list,
                          marker.name=marker.name)

    ## to initiate the parameters in recurrent event model
    para.list = list()
    beta.list = list()
    mu.list = list()
    V.list = list()
    Sigma.list = list()
    alpha.vec = rep(NA,length(data.list$marker.name))

    for(j in seq_along(alpha.vec)){

        #print(j)
        data_list_j = list(GQ_w = data.list$data.list$GQ_w,
                           GQ_t = data.list$data.list$GQ_t,
                           X = data.list$data.list$X[,j,drop=FALSE],
                           Z = data.list$data.list$Z[,j,drop=FALSE],
                           X_t = data.list$data.list$X_t[,j,drop=FALSE],
                           Z_t = data.list$data.list$Z_t[,j,drop=FALSE]
        )
        beta_j = rep(0, length(data.list$fix_est_name))
        mu_j = lapply(1:length(data.list$uni_ID), function(i){
            matrix(rep(0, length(data.list$rand_est_name)),ncol=1)
        })
        V_j = lapply(1:length(data.list$uni_ID), function(i){
            mat = matrix(0,ncol=length(data.list$rand_est_name),
                         nrow=length(data.list$rand_est_name))
            diag(mat) = 1
            mat
        })
        sigma_j = matrix(0,ncol=length(data.list$rand_est_name),
                         nrow=length(data.list$rand_est_name))
        diag(sigma_j) = 1

        para_list_j = list(beta = beta_j, mu=mu_j, V=V_j,Sigma = sigma_j)

        fit_CoxFM = init_CoxFM(data_list_j, para_list_j, 200, 1e-4)

        beta.list[[j]] = fit_CoxFM$beta
        mu.list[[j]] = fit_CoxFM$mu
        Sigma.list[[j]] = fit_CoxFM$Sigma
        V.list[[j]] = fit_CoxFM$V
        alpha.vec[j] = 0
    }

    mu.list = do.call(cbind, mu.list)

    mu_x = apply(mu.list,2, function(x){do.call(c, x)})
    # cor_m = cor(mu_x)

    V.list = do.call(cbind, V.list)

    #beta0 = runif(ncol(data.list$data.list$W_T))
    ## initiate the parameters in survival model
    Z_t = lapply(seq_along(data.list$data.list$fstat), function(i){
        list(matrix(1,length(data.list$data.list$GQ_t[[i]]),1))
    })

    Z_t = do.call(rbind, Z_t)

    Z_T = lapply(seq_along(data.list$data.list$fstat), function(i){
        list(matrix(data.list$data.list$fstat[i],1,1))
    })

    Z_T = do.call(rbind, Z_T)

    W_T = lapply(seq_along(data.list$data.list$fstat), function(i){
        list(data.list$data.list$W_T[i,,drop=FALSE]*data.list$data.list$fstat[i])
    })

    W_T = do.call(rbind, W_T)

    data_list_j = list(GQ_w = data.list$data.list$GQ_w,
                       GQ_t = data.list$data.list$GQ_t,
                       X = W_T,
                       Z = Z_T,
                       X_t = data.list$data.list$W_t,
                       Z_t = Z_t
    )

    beta_j = rep(0,ncol(data.list$data.list$W_T))
    mu_j = lapply(1:length(data.list$uni_ID), function(i){
        matrix(rep(0, 1),ncol=1)
    })
    V_j = lapply(1:length(data.list$uni_ID), function(i){
        mat = matrix(0,ncol=1,nrow=1)
        diag(mat) = 1
        mat
    })
    sigma_j = matrix(0,ncol=1, nrow=1)
    diag(sigma_j) = 1

    para_list_j = list(beta = beta_j, mu=mu_j, V=V_j,Sigma = sigma_j)

    fit_CoxFM = init_CoxFM(data_list_j, para_list_j, 200, 1e-4)
    beta0 = fit_CoxFM$beta

    para.list[["mu"]] = mu.list
    para.list[["V"]] = V.list
    para.list[["Sigma"]] = Sigma.list
    para.list[["beta"]] = beta.list
    para.list[["beta0"]] = beta0
    para.list[["alpha"]] = alpha.vec

    marker.name = data.list$marker.name
    fix_est_name = data.list$fix_est_name
    rand_est_name = data.list$rand_est_name
    surv_est_name = data.list$surv_est_name
    data.list = data.list$data.list

    init_list = list(data.list=data.list, para.list=para.list,
                     marker.name=marker.name, fix_est_name=fix_est_name,
                     rand_est_name=rand_est_name, surv_est_name=surv_est_name
    )

    init_list

}

#' @noRd
#' @keywords internal
JFM_init <- function(RecurData=NULL, SurvData = NULL,
                     control_list=NULL, marker.name=NULL){

    data.list = prep_data(RecurData=as.data.frame(RecurData),
                          SurvData = as.data.frame(SurvData),
                          control_list=control_list, marker.name=marker.name)

    ## to initiate the parameters in recurrent event model
    para.list = list()
    beta.list = list()
    mu.list = list()
    V.list = list()
    Sigma.list = list()
    alpha.vec = rep(NA,length(data.list$marker.name))

    for(j in seq_along(alpha.vec)){
        # i = 1
        #print(j)
        data_list_j = list(GQ_w = data.list$data.list$GQ_w,
                           GQ_t = data.list$data.list$GQ_t,
                           X = data.list$data.list$X[,j,drop=FALSE],
                           Z = data.list$data.list$Z[,j,drop=FALSE],
                           X_t = data.list$data.list$X_t[,j,drop=FALSE],
                           Z_t = data.list$data.list$Z_t[,j,drop=FALSE]
        )
        beta_j = rep(0, length(data.list$fix_est_name))
        mu_j = lapply(1:length(data.list$uni_ID), function(i){
            matrix(rep(0, length(data.list$rand_est_name)),ncol=1)
        })
        V_j = lapply(1:length(data.list$uni_ID), function(i){
            mat = matrix(0,ncol=length(data.list$rand_est_name),
                         nrow=length(data.list$rand_est_name))
            diag(mat) = 1
            mat
        })
        sigma_j = matrix(0,ncol=length(data.list$rand_est_name),
                         nrow=length(data.list$rand_est_name))
        diag(sigma_j) = 1

        para_list_j = list(beta = beta_j, mu=mu_j, V=V_j,Sigma = sigma_j)

        fit_CoxFM = init_CoxFM(data_list_j, para_list_j, 200, 1e-4)

        beta.list[[j]] = fit_CoxFM$beta
        mu.list[[j]] = fit_CoxFM$mu
        Sigma.list[[j]] = fit_CoxFM$Sigma
        V.list[[j]] = fit_CoxFM$V
        alpha.vec[j] = 0
    }

    mu.list = do.call(cbind, mu.list)

    mu_x = apply(mu.list,2, function(x){do.call(c, x)})
    cor_m = cor(mu_x)

    V.list = do.call(cbind, V.list)
    V.list = lapply(1:nrow(V.list), function(i){
        v=as.matrix(bdiag(V.list[i,]))
        sqrt(v) %*% cor_m %*% sqrt(v)
    })

    Sigma = as.matrix(bdiag(Sigma.list))

    ## initiate the parameters in survival model
    #beta0 = runif(ncol(data.list$data.list$W_T))
    Z_t = lapply(seq_along(data.list$data.list$fstat), function(i){
        list(matrix(1,length(data.list$data.list$GQ_t[[i]]),1))
    })

    Z_t = do.call(rbind, Z_t)

    Z_T = lapply(seq_along(data.list$data.list$fstat), function(i){
        list(matrix(data.list$data.list$fstat[i],1,1))
    })

    Z_T = do.call(rbind, Z_T)

    W_T = lapply(seq_along(data.list$data.list$fstat), function(i){
        list(data.list$data.list$W_T[i,,drop=FALSE]*data.list$data.list$fstat[i])
    })

    W_T = do.call(rbind, W_T)

    data_list_j = list(GQ_w = data.list$data.list$GQ_w,
                       GQ_t = data.list$data.list$GQ_t,
                       X = W_T,
                       Z = Z_T,
                       X_t = data.list$data.list$W_t,
                       Z_t = Z_t
    )

    beta_j = rep(0,ncol(data.list$data.list$W_T))
    mu_j = lapply(1:length(data.list$uni_ID), function(i){
        matrix(rep(0, 1),ncol=1)
    })
    V_j = lapply(1:length(data.list$uni_ID), function(i){
        mat = matrix(0,ncol=1,nrow=1)
        diag(mat) = 1
        mat
    })
    sigma_j = matrix(0,ncol=1, nrow=1)
    diag(sigma_j) = 1

    para_list_j = list(beta = beta_j, mu=mu_j, V=V_j,Sigma = sigma_j)

    fit_CoxFM = init_CoxFM(data_list_j, para_list_j, 200, 1e-4)
    beta0 = fit_CoxFM$beta


    para.list[["mu"]] = mu.list
    para.list[["V"]] = V.list
    para.list[["Sigma"]] = Sigma
    para.list[["beta"]] = beta.list
    para.list[["beta0"]] = beta0
    para.list[["alpha"]] = alpha.vec

    marker.name = data.list$marker.name
    fix_est_name = data.list$fix_est_name
    rand_est_name = data.list$rand_est_name
    surv_est_name = data.list$surv_est_name
    data.list = data.list$data.list

    init_list = list(data.list=data.list, para.list=para.list,
                     marker.name=marker.name, fix_est_name=fix_est_name,
                     rand_est_name=rand_est_name, surv_est_name=surv_est_name
    )

    init_list
}

#' @noRd
#' @keywords internal
prep_test_data <- function(RecurData_test=NULL, SurvData_test = NULL,
                           t_break = 1, tau = 0.5,
                           control_list=NULL, marker.name=NULL){

    ### control_list
    ID_name = control_list$ID_name
    item_name = control_list$item_name
    #value_name = control_list$value_name
    time_name = control_list$time_name
    fix_cov = control_list$fix_cov
    random_cov = control_list$random_cov
    recur_fix_time_fun = control_list$recur_fix_time_fun
    recur_ran_time_fun = control_list$recur_ran_time_fun
    surv_fix_time_fun = control_list$surv_fix_time_fun
    surv_time_name = control_list$surv_time_name
    surv_status_name = control_list$surv_status_name
    surv_cov =  control_list$surv_cov
    n_points =  control_list$n_points

    ### pre-process test data
    # removing subjects with event time less than t_break
    SurvData = SurvData_test[SurvData_test[,surv_time_name] > t_break, ]
    RecurData = RecurData_test[RecurData_test[,time_name] <= t_break,]
    RecurData = RecurData[is.element(RecurData[,ID_name], SurvData[,ID_name]), ]
    SurvData[,surv_time_name] = t_break
    SurvData[,surv_status_name] = 0

    ###
    data.list = list()
    para.list = list()

    fix_est_name = c(colnames(recur_fix_time_fun(1)), fix_cov)
    rand_est_name = c(colnames(recur_ran_time_fun(1)), random_cov)
    surv_est_name = c(colnames(surv_fix_time_fun(1)), surv_cov)

    ## process recurrent event data
    ## N_ij

    uni_ID = SurvData[,ID_name]
    if(is.null(marker.name)){
        marker.name = sort(unique(RecurData[,item_name]))
    }
    RecurData = RecurData[is.element(RecurData[,item_name],marker.name),]
    N_mat = sparseMatrix( i = match(RecurData[,ID_name], uni_ID),
                          j = match(RecurData[,item_name], marker.name), x= 1,
                          dims = c(length(uni_ID), length(marker.name)))
    N_mat = as.matrix(N_mat)

    ## X_i, Z_i
    X_recur_list = lapply(uni_ID, function(i){

        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        lapply(marker.name,function(x){

            time.tmp = data.tmp[data.tmp[,item_name]==x,time_name]
            if(length(time.tmp) !=0){
                time_mat.tmp = recur_fix_time_fun(time.tmp)
            }else{
                time_mat.tmp = recur_fix_time_fun(0)
                time_mat.tmp[,] = 0
            }
            if(!is.null(fix_cov)){
                cov.tmp = SurvData[SurvData[,ID_name]==i, fix_cov, drop=FALSE]
                cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])

                mat = as.matrix(cbind( time_mat.tmp,cov.tmp ))
                if(length(time.tmp) ==0){
                    mat[,] = 0
                }
                mat

            }else{
                as.matrix(cbind( time_mat.tmp ))
            }
            #cbind(1,data.tmp$years[data.tmp$item==x])
        })
    })

    X_recur_list = do.call(rbind, X_recur_list)


    Z_recur_list = lapply(uni_ID, function(i){

        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        lapply(marker.name,function(x){

            time.tmp = data.tmp[data.tmp[,item_name]==x,time_name]
            if(length(time.tmp) !=0){
                time_mat.tmp = recur_ran_time_fun(time.tmp)
            }else{
                time_mat.tmp = recur_ran_time_fun(0)
                time_mat.tmp[,] = 0
            }
            if(!is.null(random_cov)){
                cov.tmp = SurvData[SurvData[,ID_name]==i, random_cov, drop=FALSE]
                cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
                mat = as.matrix(cbind( time_mat.tmp,cov.tmp ))

                if(length(time.tmp) ==0){
                    mat[,] = 0
                }
                mat

            }else{
                as.matrix(cbind( time_mat.tmp ))
            }
            #cbind(1,data.tmp$years[data.tmp$item==x])
        })
    })

    Z_recur_list = do.call(rbind, Z_recur_list)

    ##   Z_i(T_i)
    Z_T_recur_list = lapply(uni_ID, function(i){
        T_i = SurvData[SurvData[,ID_name] == i,surv_time_name]
        #T_i = SurvData$ftime[SurvData$ID==i]
        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        lapply(marker.name,function(x){
            vv = SurvData[SurvData[,ID_name]==i, random_cov, drop=FALSE]
            matrix(c(as.numeric(recur_ran_time_fun(T_i)), as.numeric(vv[1,]) ), ncol=1)
            #matrix(c(1,T_i),ncol=1)
        })
    })
    Z_T_recur_list = do.call(rbind, Z_T_recur_list)

    ## X_i(t) , z_i(t)
    ## this depends on the number of legendre Gaussian quadrature points
    Gauss.point  = gauss.quad(n_points)
    # \int_0^{T_i} f(t)dt
    # t_node = Gauss.point$nodes *(Ti/2) + Ti/2
    # w_node = Gauss.point$weights
    # Ti/2 * sum(w_node * f(t_node))

    X_t_recur_list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        t_node = Gauss.point$nodes *(Ti/2) + Ti/2
        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        time_mat.tmp = recur_fix_time_fun(t_node)

        lapply(marker.name,function(x){
            if(!is.null(fix_cov)){
                cov.tmp = SurvData[SurvData[,ID_name]==i, fix_cov, drop=FALSE]
                cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
                as.matrix(cbind(time_mat.tmp, cov.tmp ))
            }else{
                as.matrix(cbind( time_mat.tmp ))
            }
        })
    })

    X_t_recur_list = do.call(rbind, X_t_recur_list)

    Z_t_recur_list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        t_node = Gauss.point$nodes *(Ti/2) + Ti/2
        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        time_mat.tmp = recur_ran_time_fun(t_node)

        lapply(marker.name,function(x){

            if(!is.null(random_cov)){
                cov.tmp = SurvData[SurvData[,ID_name]==i, random_cov, drop=FALSE]
                cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
                as.matrix(cbind(  time_mat.tmp[,,drop=FALSE], cov.tmp ))
            }else{
                as.matrix(cbind(time_mat.tmp[,,drop=FALSE] ))
            }
        })
    })

    Z_t_recur_list = do.call(rbind, Z_t_recur_list)


    Z_t_delta_recur_list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name] + tau
        t_node = Gauss.point$nodes *(Ti/2) + Ti/2
        data.tmp =  RecurData[RecurData[,ID_name]==i,]
        time_mat.tmp = recur_ran_time_fun(t_node)

        lapply(marker.name,function(x){

            if(!is.null(random_cov)){
                cov.tmp = SurvData[SurvData[,ID_name]==i, random_cov, drop=FALSE]
                cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
                as.matrix(cbind(  time_mat.tmp[,,drop=FALSE], cov.tmp ))
            }else{
                as.matrix(cbind(time_mat.tmp[,,drop=FALSE] ))
            }
        })
    })

    Z_t_delta_recur_list = do.call(rbind, Z_t_delta_recur_list)



    w_node.list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        Gauss.point$weights*Ti/2
    })
    t_node.list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        Gauss.point$nodes *(Ti/2) + Ti/2
    })


    w_node_delta.list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name] + tau
        Gauss.point$weights*Ti/2
    })
    t_node_delta.list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name] + tau
        Gauss.point$nodes *(Ti/2) + Ti/2
    })

    ## survival data

    ##   X_i(T_i)
    X_T_surv_list = lapply(uni_ID, function(i){
        T_i = SurvData[SurvData[,ID_name] == i,surv_time_name]
        #T_i = SurvData$ftime[SurvData$ID==i]
        #data.tmp =  RecurData[RecurData[,ID_name]==i,]
        vv = SurvData[SurvData[,ID_name]==i, surv_cov, drop=FALSE]

        c(as.numeric(surv_fix_time_fun(T_i)), as.numeric(vv[1,]) )
        #matrix(c(1,T_i),ncol=1)

    })
    X_T_surv_list = do.call(rbind, X_T_surv_list)

    ## X_i(t)
    X_t_surv_list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name]
        t_node = Gauss.point$nodes *(Ti/2) + Ti/2
        #data.tmp =  RecurData[RecurData[,ID_name]==i,]
        time_mat.tmp = surv_fix_time_fun(t_node)

        if(!is.null(surv_cov)){
            cov.tmp = SurvData[SurvData[,ID_name]==i, surv_cov, drop=FALSE]
            cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
            list(as.matrix(cbind(time_mat.tmp, cov.tmp )))
        }else{
            list(as.matrix(cbind( time_mat.tmp )))
        }

    })

    X_t_surv_list = do.call(rbind, X_t_surv_list)

    X_t_delta_surv_list = lapply(uni_ID, function(i){
        Ti = SurvData[SurvData[,ID_name] == i,surv_time_name] + tau
        t_node = Gauss.point$nodes *(Ti/2) + Ti/2
        #data.tmp =  RecurData[RecurData[,ID_name]==i,]
        time_mat.tmp = surv_fix_time_fun(t_node)

        if(!is.null(surv_cov)){
            cov.tmp = SurvData[SurvData[,ID_name]==i, surv_cov, drop=FALSE]
            cov.tmp = as.matrix(cov.tmp[rep(1,nrow(time_mat.tmp)),])
            list(as.matrix(cbind(time_mat.tmp, cov.tmp )))
        }else{
            list(as.matrix(cbind( time_mat.tmp )))
        }

    })

    X_t_delta_surv_list = do.call(rbind, X_t_delta_surv_list)

    samWt = rep(1,length(uni_ID))

    ## output list

    data.list[["N"]] = N_mat

    data.list[["X"]] = X_recur_list
    data.list[["Z"]] = Z_recur_list
    data.list[["Z_T"]] = Z_T_recur_list
    data.list[["X_t"]] = X_t_recur_list
    data.list[["Z_t"]] = Z_t_recur_list
    data.list[["Z_t_delta"]] = Z_t_delta_recur_list

    data.list[["W_T"]] = X_T_surv_list
    data.list[["W_t"]] = X_t_surv_list
    data.list[["W_t_delta"]] = X_t_delta_surv_list

    data.list[["GQ_w"]] = w_node.list
    data.list[["GQ_t"]] = t_node.list
    data.list[["GQ_w_delta"]] = w_node_delta.list
    data.list[["GQ_t_delta"]] = t_node_delta.list

    data.list[["ftime"]] = SurvData[,surv_time_name]
    data.list[["fstat"]] = SurvData[,surv_status_name]
    data.list[["samWt"]] = samWt

    list(data.list=data.list, uni_ID=uni_ID, marker.name=marker.name,
         fix_est_name=fix_est_name,rand_est_name=rand_est_name,
         surv_est_name=surv_est_name)

}

#' @noRd
#' @keywords internal
get_numH <- function(data_list=NULL, fit_obj=NULL, FUN=NULL,
                     noMUV=FALSE){
    Lvec = t(chol(fit_obj$Sigma))
    Lvec = Lvec[lower.tri(Lvec,diag=T)]
    para_MLE = c(unlist(fit_obj$beta), fit_obj$beta0,fit_obj$alpha,
                 Lvec)
    name = c(rep("beta", length(unlist(fit_obj$beta))),
             rep("beta0", length(fit_obj$beta0)),
             rep("alpha", length(fit_obj$alpha)),
             rep("L", length(Lvec)))
    hess = pracma::hessian(f=FUN, x0=para_MLE,
                           datalist=data_list, paralist=fit_obj, eps = 1e-4,noMUV=noMUV)

    npara = length(c(unlist(fit_obj$beta), fit_obj$beta0,fit_obj$alpha))
    cov = -pinv(hess)
    se = sqrt(diag(cov)[1:npara])

    # if(any(is.na(se))){
    #     hess = numDeriv::hessian(func=FUN, x=para_MLE,
    #                              datalist=data_list, paralist=fit_obj, eps = 1e-4,noMUV=noMUV)
    # }
    colnames(hess)=rownames(hess)=name
    hess
}


#' The function to get summary table of PJFM fit.
#'
#' The function is used to get summary table of PJFM fit.
#'
#' @param res a model fit returned by PJFM_fit; SE estimates are only
#' available for JFM, but not PJFM.
#'
#' @return return a data frame, which contains parameter estimates in both submodels.
#'
#' @references Jiehuan Sun. "Dynamic Prediction with Penalized Joint Frailty Model of High-Dimensional Recurrent Event Data and a Survival Outcome".
#'
PJFM_summary <- function(res=NULL){

    marker.name = res$fit$EventName
    fix_est_name = res$fit$fix_est_name
    #rand_est_name = init_list$rand_est_name
    surv_est_name = res$fit$surv_est_name

    beta.list = lapply(seq_along(marker.name), function(i){
        beta = as.numeric(res$fit$beta[[i]])
        coef_name = paste("Event_",marker.name[i],"_beta_",fix_est_name,sep="")
        names(beta) = coef_name
        beta
    })

    beta0 = as.numeric(res$fit$beta0)
    names(beta0) = paste("Surv_beta_",surv_est_name,sep="")

    alpha = as.numeric(res$fit$alpha)
    names(alpha) = paste("Event_",marker.name,"_alpha", sep="")

    para =c(do.call(c, beta.list),  beta0, alpha)
    res_summary = NULL

    if(res$object_name=="JFM"){
        cov = -pinv(res$hess)
        se = round(sqrt(diag(cov)),4)[1:length(para)]
        res_summary = data.frame(Estimate=para, SE=se,
                                 para-1.96*se, para+1.96*se)
        colnames(res_summary)[3:4] = c("95%CI_lower","95%CI_upper")
    }else{
        res_summary = data.frame(Estimate=para)
    }

    res_summary
}


#' The function to fit PJFM.
#'
#' The function is used to fit PJFM.
#'
#' @param RecurData a data frame containing the recurrent events data
#' (see \code{\link{RecurData}}).
#' @param SurvData a data frame containing the survival data
#' (see \code{\link{SurvData}}).
#' @param EventName a vector indicating which set of recurrent events
#' to be analyzed. If NULL, all recurrent events in RecurData will be used.
#' @param control_list a list of parameters specifying the joint frailty model
#' (see \code{\link{control_list}}).
#' @param nlam number of tuning parameters.
#' @param ridge ridge penalty.
#' @param pmax the maximum of biomarkers being selected.
#' The algorithm will stop early if the maximum has been reached.
#' @param min_ratio the ratio between the largest possible penalty
#' and the smallest penalty to tune.
#' @param maxiter the maximum number of iterations.
#' @param eps threshold for convergence.
#'
#' @return return a list with the following objects.
#' \item{object_name}{indicates whether this is a PJFM or JFM object. If JFM object, then some
#' recurrent events were selected and the returned model is the refitted model with
#' only selected recurrent events, but no penalty; otherwise, PJFM object is returned.}
#' \item{fit}{fitted models with estimated parameters in both submodels.}
#' \item{hess}{Hessian matrix; only available for JFM object.}
#'
#' @references Jiehuan Sun. "Dynamic Prediction with Penalized Joint Frailty Model of High-Dimensional Recurrent Event Data and a Survival Outcome".
#'
#' @examples
#'
#' require(splines)
#' data(PJFMdata)
#'
#' up_limit = ceiling(max(SurvData$ftime))
#' bs_fun <- function(t=NULL){
#'     bs(t, knots = NULL, degree = 2, intercept = TRUE, Boundary.knots= c(0,up_limit))
#' }
#'
#' recur_fix_time_fun = bs_fun
#
#' recur_ran_time_fun <- function(x=NULL){
#'     xx = cbind(1, matrix(x, ncol = 1))
#'     colnames(xx) = c("intercept","year_1")
#'     xx[,1,drop=FALSE]
#'     #xx
#' }
#'
#' surv_fix_time_fun = bs_fun
#'
#' control_list = list(
#'     ID_name = "ID", item_name = "feature_id",
#'     time_name = "time", fix_cov = "x", random_cov = NULL,
#'     recur_fix_time_fun = recur_fix_time_fun,
#'     recur_ran_time_fun = recur_ran_time_fun,
#'     surv_fix_time_fun = surv_fix_time_fun,
#'     surv_time_name = "ftime",  surv_status_name = "fstat",
#'     surv_cov = "x", n_points = 5
#' )
#'
#' \donttest{
#' ## this step takes about a few minute.
#' ## analyze the first 10 recurrent events
#' res = PJFM_fit(RecurData=RecurData, SurvData=SurvData,
#'                control_list=control_list, EventName=1:10)
#'
#' ## get summary table
#' summary_table = PJFM_summary(res)
#'
#' }
#'
PJFM_fit <- function(RecurData=NULL, SurvData = NULL,
                     control_list=NULL,EventName = NULL,
                     nlam = 50, ridge = 0, pmax = 10, min_ratio = 0.01,
                     maxiter=100, eps=1e-4){

    feature_sel = EventName

    ## initializing PJFM
    message("initializing ...")
    init_list = PJFM_init(RecurData=RecurData, SurvData = SurvData,
                          control_list=control_list, marker.name=feature_sel)

    ## run PJFM with lasso
    message("run PJFM with lasso penalty ...")
    gvec = rep(1,length(feature_sel))
    res_lasso = PJFM_covBD_seq(init_list$data.list, init_list$para.list,
                               gvec,nlam, ridge,pmax,min_ratio,maxiter,eps,
                               UseSurvN = FALSE)

    ## run PJFM with adaptive lasso
    message("run PJFM with adaptive lasso penalty ...")
    res_adalasso = NULL
    if(all(res_lasso$alpha==0)){
        res_adalasso = res_lasso
    }else{
        gvec = 1/abs(res_lasso$alpha)
        gvec[is.infinite(gvec)] = length(init_list$data.list$fstat)

        res_adalasso = PJFM_covBD_seq(init_list$data.list, init_list$para.list,
                                      gvec,nlam, ridge,pmax,min_ratio,maxiter,eps,
                                      UseSurvN = FALSE)
    }


    ## run JFM with selected predictors
    # re-create the init_list using the selected predictors
    sel_ind = which(res_adalasso$alpha!=0)
    hess = NULL
    res_final = NULL

    if(length(sel_ind)>0){
        message("run JFM with selected recurrent events ...")
        init_list = JFM_init(RecurData=RecurData, SurvData = SurvData,
                             control_list = control_list,
                             marker.name = feature_sel[sel_ind])
        init_list$para.list$beta0 = res_adalasso$beta0
        init_list$para.list$alpha = res_adalasso$alpha[sel_ind]

        res_JFM = PJFM(init_list$data.list, init_list$para.list, maxiter, eps)
        hess = get_numH(init_list$data.list, res_JFM, FUN=PJFM_numH,noMUV = FALSE)
        res_JFM = c(res_JFM,
                    list(EventName=feature_sel[sel_ind],
                         fix_est_name=init_list$fix_est_name,
                         rand_est_name=init_list$rand_est_name,
                         surv_est_name=init_list$surv_est_name)
        )
        res_final = list(object_name="JFM", fit=res_JFM, hess=hess)

    }else{
        message("No recurrent events has been selected ...")
        res_adalasso = c(res_adalasso,
                         list(EventName=feature_sel,
                              fix_est_name=init_list$fix_est_name,
                              rand_est_name=init_list$rand_est_name,
                              surv_est_name=init_list$surv_est_name)
        )
        res_final = list(object_name="PJFM", fit=res_adalasso)
    }

    res_final
}

#' The function to calculate predicted probabilities
#'
#' The function is used to calculate predicted probabilities.
#'
#' @param res a model fit returned by PJFM_fit; the prediction
#' only works the returned model fit is JFM, but not PJFM.
#' @param RecurData_test a data frame containing the recurrent events data on the test dataset
#' (see \code{\link{RecurData}}).
#' @param SurvData_test a data frame containing the survival data on the test dataset
#' (see \code{\link{SurvData}}).
#' @param control_list a list of parameters specifying the joint frailty model
#' (see \code{\link{control_list}}).
#' @param t_break the landmark time point
#' @param tau the prediction window (i.e., (t_break, t_break+tau]).
#'
#' @return return a data frame, which contains all the variables in SurvData_test as well as t_break, tau, and risk.
#' The column risk indicates the predicted probability of event in the given prediction window.
#'
#' @references Jiehuan Sun. "Dynamic Prediction with Penalized Joint Frailty Model of High-Dimensional Recurrent Event Data and a Survival Outcome".
#'
#' @examples
#' require(splines)
#' data(PJFMdata)
#'
#' up_limit = ceiling(max(SurvData$ftime))
#' bs_fun <- function(t=NULL){
#'     bs(t, knots = NULL, degree = 2, intercept = TRUE, Boundary.knots= c(0,up_limit))
#' }
#'
#' recur_fix_time_fun = bs_fun
#
#' recur_ran_time_fun <- function(x=NULL){
#'     xx = cbind(1, matrix(x, ncol = 1))
#'     colnames(xx) = c("intercept","year_1")
#'     xx[,1,drop=FALSE]
#'     #xx
#' }
#'
#' surv_fix_time_fun = bs_fun
#'
#' control_list = list(
#'     ID_name = "ID", item_name = "feature_id",
#'     time_name = "time", fix_cov = "x", random_cov = NULL,
#'     recur_fix_time_fun = recur_fix_time_fun,
#'     recur_ran_time_fun = recur_ran_time_fun,
#'     surv_fix_time_fun = surv_fix_time_fun,
#'     surv_time_name = "ftime",  surv_status_name = "fstat",
#'     surv_cov = "x", n_points = 5
#' )
#'
#' \donttest{
#' train_id = 1:200
#' test_id = 200:300
#'
#' SurvData_test = SurvData[is.element(SurvData$ID, test_id), ]
#' RecurData_test = RecurData[is.element(RecurData$ID, test_id), ]
#'
#' SurvData = SurvData[is.element(SurvData$ID, train_id), ]
#' RecurData = RecurData[is.element(RecurData$ID, train_id), ]
#'
#' ## this step takes a few minutes.
#' ## analyze the first 10 recurrent events
#' res = PJFM_fit(RecurData=RecurData, SurvData=SurvData,
#'                control_list=control_list, EventName=1:10)
#'
#'
#' ## get prediction probabilities
#' pred_scores = PJFM_prediction(res=res,RecurData_test=RecurData_test,
#'                               SurvData_test=SurvData_test,control_list=control_list,
#'                               t_break = 1, tau = 0.5)
#'
#' }
#'
PJFM_prediction <- function(res = NULL, RecurData_test=NULL, SurvData_test = NULL,
                          control_list=NULL, t_break = 1, tau = 0.5){

    res_df = NULL

    if(res$object_name!="JFM"){
        stop("No recurrent events has been selected, so no prediction is performed.")
    }else{
        fit_JFM = res$fit
        marker.name = fit_JFM$EventName

        RecurData_test= as.data.frame(RecurData_test)
        SurvData_test = as.data.frame(SurvData_test)

        ID_name = control_list$ID_name
        data_test = prep_test_data(RecurData_test= RecurData_test,
                                   SurvData_test = SurvData_test,
                                   t_break = t_break, tau = tau,
                                   control_list=control_list, marker.name=marker.name)

        fit_JFM$mu = lapply(seq_along(data_test$data.list$ftime), function(i){
            fit_JFM$mu[1,]
        })
        fit_JFM$mu = do.call(rbind, fit_JFM$mu)

        fit_JFM$V = lapply(seq_along(data_test$data.list$ftime), function(i){
            fit_JFM$V[1,]
        })
        fit_JFM$V = do.call(rbind, fit_JFM$V)


        pred_scores = PJFM_pred(data_test$data.list, fit_JFM)
        matid = match(data_test$uni_ID, SurvData_test[,ID_name])

        res_df = data.frame(SurvData_test[matid,], t_break=t_break,tau = tau,
                            risk = pred_scores)

    }

    res_df
}







