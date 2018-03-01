
# coding: utf-8

# In[ ]:


#Import libraries
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import matplotlib.pyplot as plt
import math
import PIL
import csv
from PIL import Image
from PIL import ImageOps
import numpy as np
import random


# ## Import images and labels

# In[1]:


class CelebAData:   
    #Initialize the dataset
    #@param data_size: the number of images to sample from the CelebA dataset.
    #@param training_percent: the percentage of the selected dataset to use for training (the rest is for validation)
    #@param desired_resolution: a tuple (height,width), images will be resized to this value.
    def __init__(self,data_size,training_percent,desired_resolution):
        self.data_size = data_size
        self.index=0
        self.trainSize = int(data_size*training_percent)
        labels = []
        images = []
        self.h,self.w = desired_resolution
        labels,images = self.sampleRows(data_size,labels,images)
        labels = np.asarray(labels)
        images = np.asarray(images)
        images = np.expand_dims(images, axis=3)
        self.inputdims = self.h*self.w
        print "splitting into training and test data"
        self.trainingInput,self.trainingOutput,self.testingInput,self.testingOutput = self.subsetData(labels,images)
    
    #Sample rows from the dataset. Because some rows contain corrupted data and so are ignored, 
    #this method checks to see whether enough rows have been sampled and recurses if they have not.
    def sampleRows(self,numberNeeded,labels,images):
        imagesDir = "img_align_celeba/"
        with open('list_attr_celeba.txt','rU') as f:
            lines = list(csv.reader(f,delimiter=' '))
            k = np.random.choice(np.arange(3,202600),size=numberNeeded,replace=False).tolist()
            #print "Sampling",len(k),"rows"
            print "processing images...n=",numberNeeded
            for j in k:
                line = lines[j]
                if line[16] in ('-1','1'):
                    img,label = self.parse_input(imagesDir+line[0],int(line[16]))
                    labels.append(label)
                    images.append(img)
        print "Collected",len(labels),"images,",self.data_size-len(labels),"more needed"
        if self.data_size>len(labels):
            labels,images = self.sampleRows(self.data_size-len(labels),labels,images)
        return labels,images    
    
    #Open the image file, convert to grayscale, reshape the image 
    def parse_input(self,img,label):
        if label == 1:
            one_hot = [1,0]
        else:
            one_hot = [0,1]
        try:
            new_size = (self.h,self.w)
            im = Image.open(img, 'r')
            im_grayscale = PIL.ImageOps.grayscale(im)
            im_resized = im_grayscale.resize(new_size, Image.ANTIALIAS)
            pixel_matrix = list(im_resized.getdata())
            pixel_matrix = np.array(pixel_matrix)
            pixel_matrix = pixel_matrix.reshape(new_size)

        except:
            print "could not open",img 
        return pixel_matrix,one_hot
    
    #subset data, partition this subset into training and test sets
    def subsetData(self,labels,images):
        partitionIndex = np.arange(0,self.data_size)
        np.random.seed(seed=0)
        np.random.shuffle(partitionIndex)
        trainingIndices = partitionIndex[0:self.trainSize]
        testingIndices = partitionIndex[self.trainSize:self.data_size]
        trainingInput = np.take(images,trainingIndices,axis=0)
        trainingOutput = np.take(labels,trainingIndices,axis=0)
        testingInput = np.take(images,testingIndices,axis=0)
        testingOutput = np.take(labels,testingIndices,axis=0)
        return (trainingInput,trainingOutput,testingInput,testingOutput)
    
    #Collect the next batch of data.
    def next_batch(self,inputData,outputData,batchSize):
        datalength = inputData.shape[0]
        if self.index+batchSize > datalength:
            i = self.index
            self.index = 0
            return inputData[i:datalength+1],outputData[i:datalength+1]
        else:
            i = self.index
            self.index = self.index+batchSize
            return inputData[i:i+batchSize],outputData[i:i+batchSize]


# ## Define Convolutional Neural Network

# In[2]:


class ConvolutionalNeuralNetwork:
    #Initializes the weights, intercepts, cross entropy cost function, training step size and session.
    #num_conv_layers must be in (1,2,3)
    #num_fc_layers must be in (1,2)
    def __init__(self,img_size,num_channels,num_classes,
                 batch_size,data,epochs,num_conv_layers=3,conv_filter_size=5,conv_num_filters=32,fc_layer_size=128,fc_dropout=0.,conv_dropout=0.):
        print "Training model with fc dropout of:",fc_dropout,"conv dropout of:",conv_dropout,"batch size:",batch_size,"Conv layers:",num_conv_layers,"...building model..."
        self.dataset = data
        self.figfilename = 'CNN.ImgSize'+str(img_size)+'.Epochs'+str(epochs)+'.ConvL'+str(num_conv_layers)+'.ConvFilterSize'+str(conv_filter_size)+'.ConvFilters'+str(conv_num_filters)+'.FCLSize'+str(fc_layer_size)+'.DropoutFC'+str(fc_dropout)+'.DroputConv'+str(conv_dropout)+'.png'
        num_iteration = self.dataset.trainSize/batch_size*epochs
        self.cnnsession = tf.Session()
        self.x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
        ## labels
        self.y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')
        y_true_classes = tf.argmax(self.y_true, dimension=1)
        ##Parameters:filter size and number for each layer
        filter_size_conv1 = conv_filter_size 
        num_filters_conv1 = conv_num_filters
        filter_size_conv2 = conv_filter_size
        num_filters_conv2 = conv_num_filters
        filter_size_conv3 = conv_filter_size
        num_filters_conv3 = conv_num_filters
        fc_layer_size = fc_layer_size
        if num_conv_layers == 1:
            layer_conv1 = self.create_convolutional_layer(input=self.x,
                           num_input_channels=num_channels,
                           conv_filter_size=filter_size_conv1,
                           num_filters=num_filters_conv1)
            if conv_dropout>0.:
                print "applying conv dropout of",conv_dropout
                layer_conv1_dropout = tf.nn.dropout(layer_conv1,conv_dropout,noise_shape=None,seed=None,name=None)
                layer_flat = self.create_flatten_layer(layer_conv1_dropout)
            else:
                layer_flat = self.create_flatten_layer(layer_conv1)
        if num_conv_layers == 2:
            layer_conv1 = self.create_convolutional_layer(input=self.x,
                           num_input_channels=num_channels,
                           conv_filter_size=filter_size_conv1,
                           num_filters=num_filters_conv1)
            if conv_dropout>0.:
                print "applying conv dropout of",conv_dropout
                layer_conv1_dropout = tf.nn.dropout(layer_conv1,conv_dropout,noise_shape=None,seed=None,name=None)
                layer_conv2 = self.create_convolutional_layer(input=layer_conv1_dropout,
                               num_input_channels=num_filters_conv1,
                               conv_filter_size=filter_size_conv2,
                               num_filters=num_filters_conv2)
            else:
                layer_conv2 = self.create_convolutional_layer(input=layer_conv1,
                               num_input_channels=num_filters_conv1,
                               conv_filter_size=filter_size_conv2,
                               num_filters=num_filters_conv2)
            layer_flat = self.create_flatten_layer(layer_conv2)
        if num_conv_layers == 3:
            layer_conv1 = self.create_convolutional_layer(input=self.x,
                           num_input_channels=num_channels,
                           conv_filter_size=filter_size_conv1,
                           num_filters=num_filters_conv1)
            if conv_dropout>0.:
                print "applying conv dropout of",conv_dropout
                layer_conv1_dropout = tf.nn.dropout(layer_conv1,conv_dropout,noise_shape=None,seed=None,name=None)
                layer_conv2 = self.create_convolutional_layer(input=layer_conv1_dropout,
                               num_input_channels=num_filters_conv1,
                               conv_filter_size=filter_size_conv2,
                               num_filters=num_filters_conv2)
            else:
                layer_conv2 = self.create_convolutional_layer(input=layer_conv1,
                               num_input_channels=num_filters_conv1,
                               conv_filter_size=filter_size_conv2,
                               num_filters=num_filters_conv2)
            layer_conv3= self.create_convolutional_layer(input=layer_conv2,
                           num_input_channels=num_filters_conv2,
                           conv_filter_size=filter_size_conv3,
                           num_filters=num_filters_conv3)
            layer_flat = self.create_flatten_layer(layer_conv3)
        layer_fc1 = self.create_fc_layer(input=layer_flat,
                             num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                             num_outputs=fc_layer_size,use_relu=True)
        if fc_dropout>0.:
            print "applying fc dropout of",fc_dropout
            layer_fc1_dropout = tf.nn.dropout(layer_fc1,fc_dropout,noise_shape=None,seed=None,name=None)
            layer_fc2 = self.create_fc_layer(input=layer_fc1_dropout,
                                 num_inputs=fc_layer_size,
                                 num_outputs=num_classes) 
        else:
            layer_fc2 = self.create_fc_layer(input=layer_fc1,
                                 num_inputs=fc_layer_size,
                                 num_outputs=num_classes)
        y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
        y_pred_cls = tf.argmax(y_pred, dimension=1)
        self.cnnsession.run(tf.global_variables_initializer())
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=self.y_true)
        self.cost = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        correct_prediction = tf.equal(y_pred_cls, y_true_classes)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.cnnsession.run(tf.global_variables_initializer()) 
        self.tot_iterations = 0
        saver = tf.train.Saver()
        feed_dict_val_all = {self.x: self.dataset.testingInput,
                                      self.y_true: self.dataset.testingOutput}
        val_loss_start = self.cnnsession.run(self.cost, feed_dict=feed_dict_val_all)
        val_acc_start = self.cnnsession.run(self.accuracy, feed_dict=feed_dict_val_all)
        print "About to begin training. Validation Loss:",val_loss_start,"Validation Accuracy:",val_acc_start
        self.train(num_iteration,batch_size)
        val_loss_end = self.cnnsession.run(self.cost, feed_dict=feed_dict_val_all)
        val_acc_end = self.cnnsession.run(self.accuracy, feed_dict=feed_dict_val_all)
        print "Training complete. Validation Loss:",val_loss_end,"Validation Accuracy:",val_acc_end
        
    def create_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def create_biases(self,size):
        return tf.Variable(tf.constant(0.05, shape=[size]))

    #Creates a convolutional layer and an associated max-pooling layer.
    def create_convolutional_layer(self,input,num_input_channels,conv_filter_size,num_filters):  
        #layer = tf.layers.conv2d(inputs=input,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu)

        #Define weights and biases
        weights = self.create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        biases = self.create_biases(num_filters)
        #Create the convolutional layer. Stride [1,1,1,1] means move along each dimension 1 at a time.
        layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1],padding='SAME')
        layer += biases
        #Use max_pooling
        #layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2, 2], strides=2)
        layer = tf.nn.max_pool(value=layer,ksize=[1, 3, 3, 1],strides=[1, 3, 3, 1],padding='SAME')
        #Uses rectified linear activation function
        layer = tf.nn.relu(layer)
        return layer

    def create_flatten_layer(self,layer):
        #get shape from previous layer
        layer_shape = layer.get_shape()
        #get features from previous layer
        num_features = layer_shape[1:4].num_elements()
        #reshape the number of features to a vector
        layer = tf.reshape(layer, [-1, num_features])
        return layer

    def create_fc_layer(self,input,num_inputs, num_outputs,use_relu=False):
        #Define weights and biases
        weights = self.create_weights(shape=[num_inputs, num_outputs])
        biases = self.create_biases(num_outputs)
        #multiply input*weights and add biases
        layer = tf.matmul(input, weights) + biases
        layer = tf.nn.relu(layer)
        if use_relu==True:
            layer = tf.nn.relu(layer)
        return layer
    
    #Print out the progress in terms of validation loss and accuracy and training loss and accuracy.
    def show_progress(self,epoch, feed_dict_train, feed_dict_validate, val_loss, tr_loss):
        acc = self.cnnsession.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = self.cnnsession.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0}|Training Accuracy (batch): {1:>6.1%}, Train Loss: {2:.3f}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
        print(msg.format(epoch + 1, acc, tr_loss, val_acc, val_loss))
    
    #Train the model
    #@param num_iterations the number of iterations (number of batches)
    #@param batch_size the number of batches
    #Tracks the total training iterations (train can be called multiple times without resetting)
    def train(self,num_iteration,batch_size):
        testaccuracy = []
        trainaccuracy = []
        epochtrack = []
        maxValAccuracy = 0.
        for i in range(self.tot_iterations,
                       self.tot_iterations + num_iteration):
            x_batch, y_true_batch = self.dataset.next_batch(self.dataset.trainingInput,self.dataset.trainingOutput,batch_size)
            feed_dict_tr = {self.x: x_batch,
                               self.y_true: y_true_batch}
            self.cnnsession.run(self.optimizer, feed_dict=feed_dict_tr)
            if i % int(self.dataset.trainSize/batch_size) == 0: 
                partitionIndexTrain = np.arange(0,self.dataset.trainSize)
                np.random.seed(seed=0)
                np.random.shuffle(partitionIndexTrain)
                partitionIndexValid = np.arange(0,self.dataset.data_size - self.dataset.trainSize)
                np.random.shuffle(partitionIndexValid)
                sampleIndicesTrain = partitionIndexTrain[0:500]
                sampleInputTrain = np.take(self.dataset.trainingInput,sampleIndicesTrain,axis=0)
                sampleOutputTrain = np.take(self.dataset.trainingOutput,sampleIndicesTrain,axis=0)
                sampleIndicesValid = partitionIndexValid[0:500]
                sampleInputValid = np.take(self.dataset.testingInput,sampleIndicesValid,axis=0)
                sampleOutputValid = np.take(self.dataset.testingOutput,sampleIndicesValid,axis=0)
                feed_dict_tr = {self.x: sampleInputTrain,
                                self.y_true: sampleOutputTrain}
                feed_dict_val = {self.x: sampleInputValid,
                                 self.y_true: sampleOutputValid}
                val_loss = self.cnnsession.run(self.cost, feed_dict=feed_dict_val)
                tr_loss = self.cnnsession.run(self.cost, feed_dict=feed_dict_tr)
                epoch = int(i / int(self.dataset.trainSize/batch_size))    
                acc = self.cnnsession.run(self.accuracy, feed_dict=feed_dict_tr)
                val_acc = self.cnnsession.run(self.accuracy, feed_dict=feed_dict_val)
                msg = "Training Epoch {0}|Training Accuracy (batch): {1:>6.1%}, Train Loss: {2:.3f}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
                print(msg.format(epoch + 1, acc, tr_loss, val_acc, val_loss))
                #saver.save(session, 'glasses-model') 
                epochtrack.append(epoch + 1)
                trainaccuracy.append(acc)
                testaccuracy.append(val_acc)
                if val_acc > maxValAccuracy:
                    maxValAccuracy = val_acc
        self.figfilename = str(maxValAccuracy) + self.figfilename
        print "Maximum Validation Accuracy achieved:",maxValAccuracy*100,"% at epoch",epoch+1
        self.plot(trainaccuracy,testaccuracy,epochtrack)
        self.tot_iterations += num_iteration
        
    #Plot results using matplotlib
    def plot(self,trainaccuracy,testaccuracy,epochtrack):
        plt.plot(epochtrack, testaccuracy, 'r-',linewidth=0.5)
        plt.plot(epochtrack, trainaccuracy, 'b-',linewidth=0.5)
        plt.legend(('Test accuracy', 'Train accuracy'))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig('Figures/'+self.figfilename)
        plt.show()
        plt.gcf().clear()


# ## Import datasets

# In[ ]:


#10,000 images, 80% training 20% validation
celebdata10k = CelebAData(10000,0.8,(30,30))
#20,000 images, 80% training 20% validation
celebdata20k = CelebAData(20000,0.8,(30,30))
#50,000 images, 80% training 20% validation
celebdata50k = CelebAData(50000,0.8,(30,30))


# ## Tune hyperparameters

# In[ ]:


##Test different dropout rates for both layers
dorate = [0.,0.2,0.5,0.7]
for r in dorate:
    for r2 in dorate:
        ConvolutionalNeuralNetwork(30,1,2,100,celebdata10k,100,num_conv_layers=3,fc_dropout=r,conv_dropout=r2)
##Try with 1 and 2 conv layers
for r in dorate:
    for r2 in dorate:
        ConvolutionalNeuralNetwork(30,1,2,100,celebdata10k,100,num_conv_layers=1,fc_dropout=r,conv_dropout=r2)
        ConvolutionalNeuralNetwork(30,1,2,100,celebdata10k,100,num_conv_layers=2,fc_dropout=r,conv_dropout=r2)
##Try with a different number of nodes in fc layer, keeping both dropouts at 0.5 and 3 conv layers
ConvolutionalNeuralNetwork(30,1,2,100,celebdata10k,100,num_conv_layers=3,fc_dropout=0.5,conv_dropout=0.5,fc_layer_size=64)
ConvolutionalNeuralNetwork(30,1,2,100,celebdata10k,100,num_conv_layers=3,fc_dropout=0.5,conv_dropout=0.5,fc_layer_size=256)


# ## Try with increased dataset size

# In[ ]:


ConvolutionalNeuralNetwork(30,1,2,100,celebdata20k,200,fc_dropout=0.5,conv_dropout=0.5)
ConvolutionalNeuralNetwork(30,1,2,100,celebdata50k,200,fc_dropout=0.5,conv_dropout=0.5)

