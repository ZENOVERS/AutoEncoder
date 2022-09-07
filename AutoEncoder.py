from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Activation
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

(X_tn, y_tn), (X_te, y_te) = mnist.load_data()
X_tn_re = X_tn.reshape(60000, 28, 28, 1)
X_tn = X_tn_re/255
X_te_re = X_te.reshape(10000, 28, 28, 1)
X_te = X_te_re/255

X_tn_noise = X_tn + np.random.uniform(-1, 1, size=X_tn.shape)
X_te_noise = X_te + np.random.uniform(-1, 1, size=X_te.shape)
X_tn_ns = np.clip(X_tn_noise, a_min=0, a_max=1)
X_te_ns = np.clip(X_te_noise, a_min=0, a_max=1)

input_layer1 = Input(shape=(28, 28, 1))
x1 = Conv2D(20, kernel_size=(5, 5), padding='same')(input_layer1)
x1 = Activation(activation='relu')(x1)
output_layer1 = MaxPooling2D(pool_size=2, padding='same')(x1)
encoder = Model(input_layer1, output_layer1)

input_layer2 = Input(shape=output_layer1.shape[1:4])
x2 = Conv2D(10, kernel_size=(5, 5), padding='same')(input_layer2)
x2 = Activation(activation='relu')(x2)
x2 = UpSampling2D()(x2)
x2 = Conv2D(1, kernel_size=(5, 5), padding='same')(x2)
output_layer2 = Activation(activation='relu')(x2)
decoder = Model(input_layer2, output_layer2)

input_auto = Input(shape=(28, 28, 1))
output_auto = decoder(encoder(input_auto))
auto_encoder= Model(input_auto, output_auto)

auto_encoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
hist = auto_encoder.fit(X_tn_ns, X_tn, epochs=1, batch_size=100)
X_pred = auto_encoder.predict(X_tn_ns)

plt.figure(figsize=(10, 5))
for i in range(2*5):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_pred[i].reshape(28, 28))
    
plt.show()
