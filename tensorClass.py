import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Defining a Multilayer Perceptron Model
def model(x, weights, bias):
	layer_1 = tf.add(tf.matmul(x, weights["hidden"]), bias["hidden"])
	layer_1 = tf.nn.relu(layer_1)
  
	layer_2 = tf.add(tf.matmul(layer_1, weights["hidden2"]), bias["hidden2"])
	layer_2 = tf.nn.relu(layer_2)
  
	layer_3 = tf.add(tf.matmul(layer_2, weights["hidden3"]), bias["hidden3"])
	layer_3 = tf.nn.relu(layer_3)
  
	#layer_4 = tf.add(tf.matmul(layer_3, weights["hidden4"]), bias["hidden4"])
	#layer_4 = tf.nn.relu(layer_4)
  
	output_layer = tf.matmul(layer_3, weights["output"]) + bias["output"]
  
	return output_layer


df = pd.read_csv("https://raw.githubusercontent.com/yoavalon/TensorflowClassification/master/patient.csv")
df.columns = ['bp_sys', 'bp_dy', 'oxy', 'pul', 'sug', 'cri' ]

#train_X = df[['bp_sys', 'bp_dy', 'oxy', 'pul']].iloc[0:9600,]
train_X = df[['bp_sys', 'bp_dy', 'oxy', 'pul', 'sug']].iloc[0:9600,]
train_Y = np.eye(5)[df[['cri']].iloc[0:9600,]].reshape(9600,5)

#test_X = df[['bp_sys', 'bp_dy', 'oxy', 'pul']].iloc[9600:2000,]
test_X = df[['bp_sys', 'bp_dy', 'oxy', 'pul', 'sug']].iloc[9600:2000,]
test_Y = np.eye(5)[df[['cri']].iloc[9600:10000,]].reshape(399,5)


print(train_X.shape)
print(train_Y.shape)

print(test_X.shape)
print(test_Y.shape)


#hyperparameter
learning_rate = 0.01
training_epochs = 10000
display_steps = 50


#Network parameters   #dimensions
#n_input = 4
n_input = 5
n_hidden = 10
n_output = 5

#Graph Nodes
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
		
#Weights and Biases
weights = {
	"hidden" : tf.Variable(tf.random_normal([n_input, n_hidden]), name="weight_hidden"),
	"hidden2" : tf.Variable(tf.random_normal([n_hidden, n_hidden]), name="weight_hidden2"),
	"hidden3" : tf.Variable(tf.random_normal([n_hidden, n_hidden]), name="weight_hidden3"),
	#"hidden4" : tf.Variable(tf.random_normal([n_hidden, n_hidden]), name="weight_hidden4"),
	"output" : tf.Variable(tf.random_normal([n_hidden, n_output]), name="weight_output")
}

bias = {
	"hidden" : tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden"),
  "hidden2" : tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden2"),
  "hidden3" : tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden3"),
  #"hidden4" : tf.Variable(tf.random_normal([n_hidden]), name="bias_hidden4"),
	"output" : tf.Variable(tf.random_normal([n_output]), name="bias_output")
}	

#Define model
pred = model(X, weights, bias) 

#dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

#Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#Initializing global variables
init = tf.global_variables_initializer()

lossList = []

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):		
		_, c = sess.run([optimizer, cost], feed_dict={X: train_X, Y: train_Y})		
		if(epoch + 1) % display_steps == 0:
			lossList.append(c)
			print("Epoch: ", (epoch+1), "Cost: ", c)
      
	
	test_result = sess.run(pred, feed_dict={X: train_X})
	correct_pred = tf.equal(tf.argmax(test_result, 1), tf.argmax(train_Y, 1))

	accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))
	print("Accuracy:", accuracy.eval({X: test_X, Y: test_Y}))
  
# plot Loss Graph
plt.plot(lossList)
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
