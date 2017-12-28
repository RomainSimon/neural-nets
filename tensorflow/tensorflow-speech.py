# https://github.com/llSourcell/tensorflow_speech_recognition_demo
import tflearn
import speech_data

learning_rate = 0.0001
training_iters = 300000

batch = word_batch = speech_date.mfcc_batch_generator(64)
X, Y = next (batch)
trainX, trainY = X, Y
textX, textY, X, Y

net = tflearn.input_data([None, 20, 80])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_date =learning_rate, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)

while 1:
  model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=64)
  _y=model.predict(X)
model.save('tflearn.lstm.model')
print(_y)
print(y)