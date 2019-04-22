# VISERION377 - Core
# Date : 22/04/2019
# License : MIT

import tensorflow as tf
import pre_process as ps
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.98):
            print("\nReached 98% accuracy so cancelling training!,",logs.get('acc'))
            self.model.stop_training = True

# Get preprocessed data 
(x,y) = ps.pre_process_corpus()

# Validate the training with 20% of the data
train_test_split = int(len(x)*0.20)

# Split x and y
x_train = x[train_test_split:]
y_train = y[train_test_split:]

x_test = x[:train_test_split]
y_test = y[:train_test_split]

print(len(x_train))
print(len(x_test))


### Under Dev

# model = 
callbacks = myCallback()

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
model.evaluate(x_test, y_test)

