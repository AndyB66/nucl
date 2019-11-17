import pandas as pd
import os.path
from keras import backend as K
from keras import regularizers, optimizers
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.models import load_model
from keras.optimizers import Optimizer, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import LeakyReLU
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error

train_data_path = "learn.csv"
test_data_path = 'hash.csv'
model_path = 'simple_masses_model.hdf5'
model_optimal_path = 'simple_masses_model_optimal.hdf5'
model_graph_path='simple_masses_model_graph.png'

input_dim = 4
hidden1_dim = 20
hidden2_dim = 20
output_dim = 1
num_epochs = 100000

class iRprop_(Optimizer):
    def __init__(self, init_alpha=0.01, scale_up=1.2, scale_down=0.5, min_alpha=0.00001, max_alpha=50., **kwargs):
        super(iRprop_, self).__init__(**kwargs)
        self.init_alpha = K.variable(init_alpha, name='init_alpha')
        self.scale_up = K.variable(scale_up, name='scale_up')
        self.scale_down = K.variable(scale_down, name='scale_down')
        self.min_alpha = K.variable(min_alpha, name='min_alpha')
        self.max_alpha = K.variable(max_alpha, name='max_alpha')

    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        alphas = [K.variable(K.ones(shape) * self.init_alpha) for shape in shapes]
        old_grads = [K.zeros(shape) for shape in shapes]
        self.weights = alphas + old_grads
        self.updates = []

        for p, grad, old_grad, alpha in zip(params, grads, old_grads, alphas):
            grad = K.sign(grad)
            new_alpha = K.switch(
                K.greater(grad * old_grad, 0),
                K.minimum(alpha * self.scale_up, self.max_alpha),
                K.switch(K.less(grad * old_grad, 0),K.maximum(alpha * self.scale_down, self.min_alpha),alpha)    
            )

            grad = K.switch(K.less(grad * old_grad, 0),K.zeros_like(grad),grad)
            new_p = p - grad * new_alpha 

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
            self.updates.append(K.update(alpha, new_alpha))
            self.updates.append(K.update(old_grad, grad))

        return self.updates

    def get_config(self):
        config = {
            'init_alpha': float(K.get_value(self.init_alpha)),
            'scale_up': float(K.get_value(self.scale_up)),
            'scale_down': float(K.get_value(self.scale_down)),
            'min_alpha': float(K.get_value(self.min_alpha)),
            'max_alpha': float(K.get_value(self.max_alpha)),
        }
        base_config = super(iRprop_, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# prepare input data
train_data = pd.read_csv(train_data_path).astype('float64')
test_data = pd.read_csv(test_data_path).astype('float64')

x_train = train_data.drop(columns = ['BE/A(MeV)', 'przewidywane_BE/A(MeV)'])
y_train = train_data[['BE/A(MeV)']]

x_test = test_data.drop(columns = ['BE/A(MeV)', 'przewidywane_BE/A(MeV)'])
y_test = test_data[['BE/A(MeV)']]

my_batch = train_data.shape[0]
#my_batch = 32

#print(train_data.shape)
#print(test_data.shape)
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

if os.path.exists(model_optimal_path) and os.path.isfile(model_optimal_path):
    #load model from file
    print("Loading model...")
    model = load_model(model_optimal_path, custom_objects={'iRprop_': iRprop_})
else:
    #create model
    model = Sequential()

    #add model layers
    model.add(Dense(hidden1_dim, activation='sigmoid', input_shape=(input_dim,)))
#    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(hidden2_dim, activation='sigmoid'))
    model.add(Dense(output_dim, activation='linear'))
    plot_model(model, to_file=model_graph_path, show_shapes=True, dpi=300)

#compile model using mse as a measure of model performance
model.compile(optimizer=iRprop_(), loss='mean_squared_error')
#model.compile(optimizer='nadam', loss='mean_squared_error')

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=50, monitor='loss')

#train model
model.fit(x_train, y_train, 
          batch_size = my_batch,
          epochs=num_epochs, 
          shuffle=False, 
          validation_data=(x_test, y_test),
          callbacks=[early_stopping_monitor])

# save model
model.save(model_path)

#make predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
pd.set_option("display.max_rows", 10)
print(abs(y_train_pred - y_train).sort_values(by='BE/A(MeV)'))
print(abs(y_test_pred - y_test).sort_values(by='BE/A(MeV)'))

print('MSQE(train)', mean_squared_error(y_train, y_train_pred))
print('MSQE(test) ', mean_squared_error(y_test, y_test_pred))

K.clear_session()
