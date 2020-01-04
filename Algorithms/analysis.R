setwd(dirname(rstudioapi::getSourceEditorContext()$path))

find_l_reg <- function(path, dataset){
  my_files <- list.files(path)
  my_files <- my_files[grepl(paste(dataset, ".*.txt", sep=""), my_files)]
  lambdas <- unlist(unique(lapply(my_files, FUN=function(x){ unlist(strsplit(unlist(strsplit(x, "_"))[3], "l"))[2] })))
  return(lambdas)
}

plot_res <- function(path, old_path, l_reg, dataset, metric){
  
  l_reg_null <- FALSE
  if(is.null(l_reg)){
    l_reg <- find_l_reg(old_path, dataset)
    l_reg_null <- TRUE
  }
  
  all_res <- c() # AE LLE _ X encoded
  all_res2 <- c() # AE LLE _ Y of LLE
  all_res_ae <- c()
  all_res_lle <- c()
  for(l in l_reg){
    if(l_reg_null)
      met <- read.csv(paste(old_path, dataset, "_5_l", l, "_res_", metric, ".txt", sep=""), header=F)
    else
      met <- read.csv(paste(old_path, dataset, "_5_l", l, "_ae_lle_res_", metric, ".txt", sep=""), header=F)
    res <- mean(met[,3])
    all_res <- cbind(all_res, res)
    res <- mean(met[,4])
    all_res2 <- cbind(all_res2, res)
    
    all_res_ae <- c(all_res_ae, met[,1])
    all_res_lle <- c(all_res_lle, met[,2])
  }
  if(l_reg_null){
    l <- find_l_reg(path, dataset)[1]
    met <- read.csv(paste(path, dataset, "_5_l", l, "_res_", metric, ".txt", sep=""), header=F)
  }else
    met <- read.csv(paste(path, dataset, "_5_ae_lle_res_", metric, ".txt", sep=""), header=F)
  res <- mean(met[,3])
  all_res <- cbind(res, all_res)
  res <- mean(met[,4])
  all_res2 <- cbind(res, all_res2)
  
  if(l_reg_null){
    new_names <- c(l, l_reg)

    my_order <- order(as.numeric(new_names))
    all_res <- all_res[1, my_order]
    all_res2 <- all_res2[1, my_order]
    new_names <- new_names[my_order]
    
    names(all_res) <- new_names
    names(all_res2) <- new_names
  }else{
    colnames(all_res) <- c("1.0", l_reg)
    colnames(all_res2) <- c("1.0", l_reg) 
    
    all_res <- t(all_res)
    all_res2 <- t(all_res2)
  }
  
  # Plot
  plot(all_res, xaxt = "n", type = "b", ylab = "", xlab = "Lambda", ylim=c(0,1), main=paste(dataset, metric))
  lines(all_res2, pch = 2, lty = 4, type="b")
  abline(h=mean(all_res_ae), lty = 2)
  abline(h=mean(all_res_lle), lty = 3)
  if(l_reg_null)
    axis_labels <- names(all_res)
  else
    axis_labels <- colnames(all_res)
  axis(1, at=1:length(axis_labels), labels=axis_labels)
  legend("topright",legend=c("X encoded", "Y (LLE)"), lty = c(1,4), pch = 1:2, cex = 0.75)
  #legend("bottomright",legend=c("X encoded", "Y (LLE)"), lty = c(1,4), pch = 1:2, cex = 0.75)
}

plot_res_both <- function(path, old_path, dataset, l_reg=NULL){
  layout(matrix(1:2, nrow=1))
  plot_res(path, old_path, l_reg, dataset, "nmi")
  plot_res(path, old_path, l_reg, dataset, "ari")
}

# path <- "../Results/Images/"
# old_path <- "../Results/Images/old/"
# l_reg <- c("2.0", "10.0", "20.0", "50.0")
# dataset <- "ORL"

path <- "../Results/FCPS_AE_LLE/"
old_path <- "../Results/FCPS_AE_LLE/old2/"
l_reg <- NULL
dataset <- "EngyTime"

plot_res_both(path, old_path, dataset, l_reg=l_reg)













#### OLD


plot_res <- function(path, old_path, l_reg, dataset, metric){
  all_res <- c() # AE LLE _ X encoded
  all_res2 <- c() # AE LLE _ Y of LLE
  all_res_ae <- c()
  all_res_lle <- c()
  for(l in l_reg){
    met <- read.csv(paste(old_path, dataset, "_5_l", l, "_ae_lle_res_", metric, ".txt", sep=""), header=F)
    res <- mean(met[,3])
    all_res <- cbind(all_res, res)
    res <- mean(met[,4])
    all_res2 <- cbind(all_res2, res)
    all_res_ae <- cbind(all_res_ae, met[,1])
    all_res_lle <- cbind(all_res_lle, met[,2])
  }
  met <- read.csv(paste(path, dataset, "_5_ae_lle_res_", metric, ".txt", sep=""), header=F)
  res <- mean(met[,3])
  all_res <- cbind(res, all_res)
  res <- mean(met[,4])
  all_res2 <- cbind(res, all_res2)
  
  colnames(all_res) <- c("1.0", l_reg)
  colnames(all_res2) <- c("1.0", l_reg)
  
  # Plot
  plot(t(all_res), xaxt = "n", type = "b", ylab = "", xlab = "Lambda", ylim=c(0,1), main=paste(dataset, metric))
  lines(t(all_res2), pch = 2, lty = 4, type="b")
  abline(h=mean(all_res_ae), lty = 2)
  abline(h=mean(all_res_lle), lty = 3)
  axis(1, at=1:length(colnames(all_res)), labels=colnames(all_res))
  legend("topright",legend=c("X encoded", "Y (LLE)"), lty = c(1,4), pch = 1:2, cex = 0.75)
}