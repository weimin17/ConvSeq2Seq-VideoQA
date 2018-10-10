from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import pandas as pd

from tensorflow.contrib.layers import fully_connected
from random import randint
from collections import namedtuple

# parameters
n_words = 7088 # number of words
batch_size = 100
length_enc1 = 50 # num steps for encoding features
length_enc2 = 15 # num steps for encoding question embeddings
epoch_num = 6000 # training epoches
learning_rate = 0.0001
input_size = 1000
output_size = 1000

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("cnn_layers", 4, "Number of layers in encoder.")
tf.flags.DEFINE_string("cnn_nhids", "256,256,256,256", "Number of units in each layer.")
tf.flags.DEFINE_string("cnn_kwidths", "3,3,3,3", "Kernel size of each layer.")
tf.flags.DEFINE_integer("cnn_nhid_default", 256, "Default size of hidden layer.")
tf.flags.DEFINE_integer("cnn_kwidth_default", 3, "Default size of kernel.")
tf.flags.DEFINE_float("embedding_dropout_keep_prob", 0.9, "Dropout keep probability applied to the embeddings.")
tf.flags.DEFINE_float("nhid_dropout_keep_prob", 0.9, "Dropout keep probability applied to the hidden layers.")

EncoderOutput = namedtuple("EncoderOutput", "outputs final_state attention_values")

def parse_list_or_default(params_str, number, default_val, delimitor = ','):
	param_list = []
	if params_str == "":
		param_list = [default_val] * number
	else:
		param_list = [int(x) for x in params_str.strip().split(delimitor)]
	return param_list

def linear_mapping_weightnorm(inputs, out_dim, in_dim = None, dropout = 1.0, var_scope_name = "linear_mapping"):
	# inputs --> [batch_size, 50, 1000]
	# out_dim --> 256
	# in_dim = None
	# dropout = 0.9
	with tf.variable_scope(var_scope_name):
		input_shape = inputs.get_shape().as_list() # [batch_size, 50, 1000]
		input_shape_tensor = tf.shape(inputs) # value is [batch_size, 50, 1000] with shape (3,)
		
		# use weight normalization (Salimans & Kingma, 2016)  w = g* v/2-norm(v)
		V = tf.get_variable('V', shape = [int(input_shape[-1]), out_dim], dtype = tf.float32, initializer = tf.random_normal_initializer(mean = 0, stddev = tf.sqrt(dropout*1.0/int(input_shape[-1]))), trainable = True)
		# V --> [1000, 256]
		V_norm = tf.norm(V.initialized_value(), axis = 0) # V shape is M*N,  V_norm shape is N
		# V_norm --> [256]
		g = tf.get_variable('g', dtype = tf.float32, initializer = V_norm, trainable = True) # g --> [256]
		b = tf.get_variable('b', shape = [out_dim], dtype = tf.float32, initializer = tf.zeros_initializer(), trainable = True) # b --> [256]
	
		assert len(input_shape) == 3
		inputs = tf.reshape(inputs, [-1, input_shape[-1]]) # [batch_size, 50, 1000] --> [batch_size * 50, 1000]
		inputs = tf.matmul(inputs, V) # [batch_size * 50, 1000] x [1000, 256] --> [batch_size * 50, 256]
		inputs = tf.reshape(inputs, [input_shape_tensor[0], -1, out_dim]) # [batch_size, 50, 256]
		#inputs = tf.matmul(inputs, V) # x*v
	
		scaler = tf.div(g, tf.norm(V, axis = 0)) # g/2-norm(v) --> [256]
		inputs = tf.reshape(scaler, [1, out_dim]) * inputs + tf.reshape(b, [1, out_dim])   # x*v g/2-norm(v) + b
		return inputs # [batch_size, 50, 256]

def conv1d_weightnorm(inputs, layer_idx, out_dim, kernel_size, padding = "SAME", dropout = 1.0,  var_scope_name = "conv_layer"): #padding should take attention
	with tf.variable_scope("conv_layer_" + str(layer_idx)):
		in_dim = int(inputs.get_shape()[-1])
		V = tf.get_variable('V', shape = [kernel_size, in_dim, out_dim], dtype = tf.float32, initializer = tf.random_normal_initializer(mean = 0, stddev = tf.sqrt(4.0*dropout/(kernel_size*in_dim))), trainable = True)
		V_norm = tf.norm(V.initialized_value(), axis = [0,1])  # V shape is M*N*k,  V_norm shape is k  
		g = tf.get_variable('g', dtype = tf.float32, initializer = V_norm, trainable = True)
		b = tf.get_variable('b', shape = [out_dim], dtype = tf.float32, initializer = tf.zeros_initializer(), trainable = True)
	
		# use weight normalization (Salimans & Kingma, 2016)
		W = tf.reshape(g, [1, 1, out_dim]) * tf.nn.l2_normalize(V, [0,1])
		inputs = tf.nn.bias_add(tf.nn.conv1d(value = inputs, filters = W, stride = 1, padding = padding), b)   
		return inputs

def gated_linear_units(inputs):
	input_shape = inputs.get_shape().as_list()
	assert len(input_shape) == 3
	input_pass = inputs[:, :, 0 : int(input_shape[2]/2)]
	input_gate = inputs[:, :, int(input_shape[2]/2):]
	input_gate = tf.sigmoid(input_gate)
	return tf.multiply(input_pass, input_gate)

def V_feature_conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict, mode):
	# inputs --> [batch_size, 50, 256]
	# nhids_list --> [256, 256, 256, 256]
	# kwidths_list --> [3, 3, 3, 3]
	# dropout_dict --> {'src': 0.9, 'hid': 0.9}

	next_layer = inputs # [batch_size, 50, 256]
	for layer_idx in range(len(nhids_list)): # [0, 1, 2, 3]
		nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1] # first layer nin = nhids_list[layer_idx], other layers nin = nhids_list[layer_idx - 1]
		nout = nhids_list[layer_idx] # nout = nhids_list[layer_idx]
		if nin != nout:
			# mapping for res add
			res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout = dropout_dict['src'], var_scope_name = "linear_mapping_cnn_" + str(layer_idx)) # [batch_size, 50, 256] 
		else:
			res_inputs = next_layer # [batch_size, 50, 256]

		# dropout before input to conv
		next_layer = tf.contrib.layers.dropout(inputs = next_layer, keep_prob = dropout_dict['hid'], is_training = mode) # [batch_size, 50, 256]
		next_layer = conv1d_weightnorm(inputs = next_layer, layer_idx = layer_idx, out_dim = nout * 2, kernel_size = kwidths_list[layer_idx], padding = "SAME", dropout = dropout_dict['hid'], var_scope_name = "conv_layer_" + str(layer_idx)) 
		next_layer = gated_linear_units(next_layer) # [batch_size, 50, 256]
		next_layer = (next_layer + res_inputs) * tf.sqrt(0.5) # [batch_size, 50, 256]
	return next_layer # [batch_size, 50, 256]

def Q_emb_conv_encoder_stack(inputs, nhids_list, kwidths_list, dropout_dict, mode):
	# inputs --> [batch_size, 15, 256]
	# nhids_list --> [256, 256, 256, 256]
	# kwidths_list --> [3, 3, 3, 3]
	# dropout_dict --> {'src': 0.9, 'hid': 0.9}

	next_layer = inputs # [batch_size, 15, 256]
	for layer_idx in range(len(nhids_list)): # [0, 1, 2, 3]
		nin = nhids_list[layer_idx] if layer_idx == 0 else nhids_list[layer_idx - 1] # first layer nin = nhids_list[layer_idx], other layers nin = nhids_list[layer_idx - 1]
		nout = nhids_list[layer_idx] # nout = nhids_list[layer_idx]
		if nin != nout:
			# mapping for res add
			res_inputs = linear_mapping_weightnorm(next_layer, nout, dropout = dropout_dict['src'], var_scope_name = "linear_mapping_cnn_" + str(layer_idx)) # [batch_size, 50, 256] 
		else:
			res_inputs = next_layer # [batch_size, 15, 256]

		# dropout before input to conv
		next_layer = tf.contrib.layers.dropout(inputs = next_layer, keep_prob = dropout_dict['hid'], is_training = mode) # [batch_size, 15, 256]
		next_layer = conv1d_weightnorm(inputs = next_layer, layer_idx = layer_idx, out_dim = nout * 2, kernel_size = kwidths_list[layer_idx], padding = "SAME", dropout = dropout_dict['hid'], var_scope_name = "conv_layer_" + str(layer_idx)) 
		next_layer = gated_linear_units(next_layer) # [batch_size, 15, 256]
		next_layer = (next_layer + res_inputs) * tf.sqrt(0.5) # [batch_size, 15, 256]
	return next_layer # [batch_size, 15, 256]

def V_feature_encode(input_V_feature, mode):

	embed_size = input_V_feature.get_shape().as_list()[-1] # (batch_size, 50, 1000) --> [batch_size, 50, 1000] --> 1000

	# Apply dropout to embeddings
	input_V_feature = tf.contrib.layers.dropout(inputs = input_V_feature, keep_prob = FLAGS.embedding_dropout_keep_prob, is_training = mode) # [batch_size, 50, 1000] --> [batch_size, 50, 1000]
	
	with tf.variable_scope("V_feature_encoder_cnn"):    
		next_layer = input_V_feature # [batch_size, 50, 1000]
		if FLAGS.cnn_layers > 0: # 4 > 0
			nhids_list = parse_list_or_default(FLAGS.cnn_nhids, FLAGS.cnn_layers, FLAGS.cnn_nhid_default) # [256, 256, 256, 256]
			kwidths_list = parse_list_or_default(FLAGS.cnn_kwidths, FLAGS.cnn_layers, FLAGS.cnn_kwidth_default) # [3, 3, 3, 3]
		
			# mapping emb dim to hid dim
			next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0], dropout = FLAGS.embedding_dropout_keep_prob, var_scope_name = "linear_mapping_before_cnn")      
			# next_layer --> [batch_size, 50, 256]
			next_layer = V_feature_conv_encoder_stack(next_layer, nhids_list, kwidths_list, {'src': FLAGS.embedding_dropout_keep_prob, 'hid': FLAGS.nhid_dropout_keep_prob}, mode)
			# next_layer --> [batch_size, 50, 256]
			next_layer = linear_mapping_weightnorm(next_layer, embed_size, var_scope_name = "linear_mapping_after_cnn")
			# next_layer --> [batch_size, 50, 1000]
		## The encoder stack will receive gradients *twice* for each attention pass: dot product and weighted sum.
		##cnn = nn.GradMultiply(cnn, 1 / (2 * nattn))  
		cnn_c_output = (next_layer + input_V_feature) * tf.sqrt(0.5) # [batch_size, 50, 1000]
			
	final_state = tf.reduce_mean(cnn_c_output, 1) # [batch_size, 50, 1000] --> [batch_size, 1000]
	return EncoderOutput(outputs = next_layer, final_state = final_state, attention_values = cnn_c_output)

def Q_emb_encode(input_Q_emb, mode):

	embed_size = input_Q_emb.get_shape().as_list()[-1] # (batch_size, 15, 1000) --> [batch_size, 15, 1000]--> 1000

	# Apply dropout to embeddings
	input_Q_emb = tf.contrib.layers.dropout(inputs = input_Q_emb, keep_prob = FLAGS.embedding_dropout_keep_prob, is_training = mode) # [batch_size, 15, 1000] --> [batch_size, 15, 1000]
	
	with tf.variable_scope("Q_emb_encoder_cnn"):
		next_layer = input_Q_emb # [batch_size, 15, 1000]
		if FLAGS.cnn_layers > 0: # 4 > 0
			nhids_list = parse_list_or_default(FLAGS.cnn_nhids, FLAGS.cnn_layers, FLAGS.cnn_nhid_default) # [256, 256, 256, 256]
			kwidths_list = parse_list_or_default(FLAGS.cnn_kwidths, FLAGS.cnn_layers, FLAGS.cnn_kwidth_default) # [3, 3, 3, 3]
		
			# mapping emb dim to hid dim
			next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0], dropout = FLAGS.embedding_dropout_keep_prob, var_scope_name = "linear_mapping_before_cnn")      
			# next_layer --> [batch_size, 15, 256]
			next_layer = Q_emb_conv_encoder_stack(next_layer, nhids_list, kwidths_list, {'src': FLAGS.embedding_dropout_keep_prob, 'hid': FLAGS.nhid_dropout_keep_prob}, mode)
			# next_layer --> [batch_size, 15, 256]
			next_layer = linear_mapping_weightnorm(next_layer, embed_size, var_scope_name = "linear_mapping_after_cnn")
			# next_layer --> [batch_size, 15, 1000]
		
		## The encoder stack will receive gradients *twice* for each attention pass: dot product and weighted sum.
		##cnn = nn.GradMultiply(cnn, 1 / (2 * nattn))  
		cnn_c_output = (next_layer + input_Q_emb) * tf.sqrt(0.5) # [batch_size, 15, 1000]
			
	final_state = tf.reduce_mean(cnn_c_output, 1) # [batch_size, 15, 1000] --> [batch_size, 1000]
	return EncoderOutput(outputs = next_layer, final_state = final_state, attention_values = cnn_c_output)


def make_attention(output_Q_emb, output_V_feature):
	with tf.variable_scope("attention_layer"):

		output_Q_emb_a = output_Q_emb.outputs # [batch_size, 15, 1000]
		output_Q_emb_c = output_Q_emb.attention_values # [batch_size, 15, 1000]
		output_V_feature_a = output_V_feature.outputs # [batch_size, 50, 1000]
		output_V_feature_c = output_V_feature.attention_values # [batch_size, 50, 1000]
	 
		att_score = tf.matmul(output_Q_emb_a, output_V_feature_a, transpose_b = True) # [batch_size, 15, 1000] ** [batch_size, 50, 1000] --> [batch_size, 15, 50]
		att_score = tf.nn.softmax(att_score) # [batch_size, 15, 50]
  
		length = tf.cast(tf.shape(output_V_feature_c), tf.float32) # [batch_size, 50, 1000]
		att_out = tf.matmul(att_score, output_V_feature_c) * length[1] * tf.sqrt(1.0/length[1]) # [batch_size, 15, 50] ** [batch_size, 50, 1000] --> [batch_size, 15, 1000]
	 
		att_out = linear_mapping_weightnorm(att_out, output_Q_emb_a.get_shape().as_list()[-1], var_scope_name = "linear_mapping_att_out") # [batch_size, 15, 1000]
	return att_out # [batch_size, 15, 1000]


def VQA_model(input_V_feature, input_Q_emb, mode):
	# input_V_feature = [batch_size, 50, 4096]
	# input_Q_emb = [batch_size, 15, 7088]

	# feature embedding
	input_V_feature = tf.reshape(input_V_feature, [-1, 4096]) # [batch_size, 50, 4096] --> [batch_size x 50, 4096]
	f1 = fully_connected(input_V_feature, num_outputs = input_size) # [batch_size x 50, 4096] --> [batch_size x 50, 1000]
	input_V_feature = tf.reshape(f1, [-1, length_enc1, input_size]) # [batch_size x 50, 1000] --> [batch_size, 50, 1000]

	# word embedding
	input_Q_emb = tf.reshape(input_Q_emb, [-1, n_words]) # [batch_size, 15, 7088] --> [batch_size x 15, 7088]
	hidden_Q_emb1 = {'weights':tf.Variable(tf.random_normal([n_words, input_size])), 'biases':tf.Variable(tf.random_normal([input_size]))}
	input_Q_emb = tf.add(tf.matmul(input_Q_emb, hidden_Q_emb1['weights']), hidden_Q_emb1['biases']) # [batch_size x 15, 1000] 
	input_Q_emb = tf.reshape(input_Q_emb, [-1, length_enc2, input_size]) # [batch_size, 15, 1000]

	# conv encode visual feature
	output_V_feature = V_feature_encode(input_V_feature, mode)
	output_Q_emb = Q_emb_encode(input_Q_emb, mode)
	output_emb = make_attention(output_Q_emb, output_V_feature) # [batch_size, 15, 1000]
	output_final = tf.reduce_mean(output_emb, 1) # [batch_size, 15, 1000] --> [batch_size, 1000]

	# DNN
	hidden_Q_emb2 = {'weights':tf.Variable(tf.random_normal([output_size, n_words])), # [output_size, n_words] --> [1000, 7088]
					  'biases':tf.Variable(tf.random_normal([n_words]))}
	output = tf.add(tf.matmul(output_final, hidden_Q_emb2['weights']), hidden_Q_emb2['biases']) # [batch_size, 7088]
	# output = tf.nn.sigmoid(output)
	# print('output' + '\n', output)

	return output # [batch_size, 7088]

# Input and Label
X_enc_1 = tf.placeholder("float", [None, 50, 4096], name = 'X_enc_1') # video features = [batch_size, 50, 4096]
X_enc_2 = tf.placeholder("float", [None, 15, n_words], name = 'X_enc_2') # question embeddings = [batch_size, 15, 7088]
Y = tf.placeholder("float", [None, n_words], name = 'Y') # answer embeddings = [batch_size, 7088]

# Build Graph
prediction = VQA_model(X_enc_1, X_enc_2, True) # [batch_size, 7088]
labels = Y # [batch_size, 7088]

with tf.name_scope("cost_func") as scope:
	cross_entropy_cost = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = prediction)
	cost_sum = tf.reduce_sum(cross_entropy_cost)
	tf.summary.scalar("cost_func", cost_sum)

train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_sum)
# train_op = tf.train.MomentumOptimizer(learning_rate = 0.25, momentum = 0.99, use_nesterov = True).minimize(cost_sum)
# train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost_sum)

# Training Process
saver = tf.train.Saver(max_to_keep = 120)
if not os.path.exists('./Models_Convseq2seq'):
	os.makedirs('./Models_Convseq2seq')

# Launch the graph in a session
with tf.Session() as sess:
	tf.global_variables_initializer().run()

	# Merge all summaries into a single operator
	merged_summary_op = tf.summary.merge_all()
	if not os.path.exists('./Tensorboard/Models_Convseq2seq'):
		os.makedirs('./Tensorboard/Models_Convseq2seq')
	summary_writer = tf.summary.FileWriter('./Tensorboard/Models_Convseq2seq', graph_def = sess.graph_def)
	
	for i in range(epoch_num): # 1 Epoch = 10 batches training

		batch_cost_sum = 0
		for j in range(10): # 10 batches
			index = randint(1, 180)
			# print(index)
			input_feature = np.load('../data/Batch_data/Feature/feature_batch_' + str(index) + '.npy') # [batch_size, 50, 4096]
			input_Q_emb = np.load('../data/Batch_data/Q/Q_batch_' + str(index) + '.npy') # [batch_size, 15, 7088]
			input_A_emb = np.load('../data/Batch_data/A/A_batch_' + str(index) + '.npy') # [batch_size, 7088]
			cost_batch, _ = sess.run([cost_sum, train_op], feed_dict = {X_enc_1: input_feature, X_enc_2: input_Q_emb, Y: input_A_emb})
			batch_cost_sum = batch_cost_sum + cost_batch

		print('Epoch', i, ':', 'Average Sum Cost of 10 Batches (Train Dataset)', batch_cost_sum / 10)

		if i % 50 == 0:
			save_path = saver.save(sess, './Models_Convseq2seq/model.ckpt_' + str(i))
			print("Model saved in file: %s" % save_path)

