import dataset
import Plot
import FormalizeData as fd
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D,MaxPooling1D, Embedding, BatchNormalization, Bidirectional
from keras.callbacks import History, ModelCheckpoint

numpy.random.seed()

#Initiliaze
history = History()
X_train_seqs,y_train,X_test_seqs,y_test = dataset.load_data()
X_train,y_train,X_test,y_test = dataset.integer_region_encode(X_train_seqs,y_train,X_test_seqs,y_test)

# create the model
model = Sequential()
model.add(Embedding(input_dim=4096,output_dim=64, input_length=497))
model.add(Dropout(0.05))
model.add(Conv1D(filters=16, kernel_size=51, padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.6))
model.add(Conv1D(filters=8, kernel_size=51, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.6))
model.add(Bidirectional(LSTM(units=100)))
model.add(Dropout(0.25))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#checkpoints
filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')
callbacks_list = [checkpoint,history]

print(model.summary())

model.fit(X_train,y_train, validation_data=(X_test,y_test), nb_epoch=1200, batch_size=64, callbacks=callbacks_list, verbose=2)

dict = history.history
loss_list_training = dict['loss']
acc_list_training = dict['acc']
loss_list_validation = dict['val_loss']
acc_list_validation = dict['val_acc']

# Final evaluation of the model

model.load_weights("weights.best.hdf5")
# Compile model (required to make predictions)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

for i in range(0,len(scores)):
    print(scores[i])

# Predict
pred_results = model.predict_classes(X_test)

#Formalize Data
fd.formalize_data(X_train_seqs,y_train,X_test_seqs,y_test,14963,64,scores[1],pred_results)

#Plots
Plot.plot_graph_training(loss_list_training,acc_list_training,1200)
Plot.plot_graph_validation(loss_list_validation,acc_list_validation,1200)
Plot.plot_bothAccuracies(acc_list_training,acc_list_validation,1200)
Plot.plot_bothLosses(loss_list_training,loss_list_validation,1200)