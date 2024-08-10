rm(list = ls())
setwd(this.path::here())
library(keras)
library(tensorflow)

load("../code-RKeras/simulated-data/qregress_train_df.Rdata")

X<- X_train[1:9000,,,]
Y<- Y_train[1:9000,,]
X_valid<- X_train[9001:10000,,,]
Y_valid<- Y_train[9001:10000,,]

input.lay <- layer_input(shape = dim(X)[2:(length(dim(X)))], name = 'input_layer')



tilted_loss <- function(  y_true, y_pred) {
  #Here's the backend functions from Keras
  K<-backend()
  e = (y_true - y_pred)
  tau <- .9
  #e = (y-f)
  #return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)
  return(K$mean(K$maximum(tau*e,(tau-1)*e)))
}


hidden.lay <- input.lay %>%
  layer_dense(units=5,activation = 'relu', name = 'hidden_1') %>%
  layer_dense(units=5,activation = 'relu', name = 'hidden_2') %>%
  layer_dense(units=5,activation = 'relu', name = 'hidden_3')  %>%
  layer_dense(units=5,activation = 'relu', name = 'hidden_4')


output.lay <- hidden.lay %>%
  layer_dense(units=1,activation = 'linear', name = 'output') 


model <- keras_model(
  inputs = c(input.lay), 
  outputs = c(output.lay)
)

model %>% compile(
  optimizer="adam",
  loss = tilted_loss #And here we specify our loss function
)
#We will use early stopping to avoid overfitting.
stop <- callback_early_stopping( monitor = "val_loss", 
                                 patience = 50,
                                 verbose = 0,
                                 mode = "min", 
                                 restore_best_weights = TRUE)
history <- model %>% fit(
  x=X_train, y=Y_train,
  epochs = 2500, batch_size = 1000,
  callback=stop,
  validation_data = list(X_valid,Y_valid)
)



pred1<-predict(model, X_test)
tilted_loss(Y_test, pred1[,,,1])

#tilted_loss(y_true=Y_test[], pred1)
########################################## 
############## GPD loss ###################
############## ############## ############## 

exceed=Y_tain-pred_u_train
inds= exceeds>0
exceeds[inds]

new.X_Train=matrix(nrow = length(inds), ncol = dim(X_train)[4])
for( i in 1:dim(X_train)){
  new.X_Train[,i]<- X_train[,,,i][inds]
}

gpd_loss<- function(y_pred, y_true){
  
  
}


