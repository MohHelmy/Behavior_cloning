# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Network_Architecture.png "Network Architecture"
[image2]: ./examples/model.png "Model Architecture"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The Nivida Model is used as a startign point.

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting and increase my model's ability to generalize for driving on unseen roads, I artificially increased my dataset using a couple of proven image augmentation techniques.

* Adjusting the brightness of the images.
* Scaling up or down the V channel by a random factor
* Cropping from the top and the bottom from each image in order to remove any noise from the sky or trees in the top of the * * * Images and the car's hood from the bottom of the image. (line 123: model.add(Cropping2D(cropping = ((74,25), (0,0)),input_shape=(160, 320, 3))) )

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road to train the module how to recover from going far left or far right.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The solution is based on NVIDIA's model.

One fully connected layer was added producing one output that is used for steering. Also, the input layer was subjected to normalization and cropping to reduce unnecessary processing and memory resources.

Front-facing center camera image and steering angle data were split into a training and validation set. Training using adam optimizer with the mean square error. 

To help the model generalize, the right and left camera images were used with steering angle adjusted randomly between 0.01-0.2 so that the model would generalize and recover the vehicle from the side of the road. 

Training with the extended data set of 19284 images over 10 epochs resulted in 1.39% validation loss.

Running the simulator to see how well the car was driving around track one, the vehicle is able to drive autonomously around the track without leaving the road (see video.mp4). 


The final step was to run the simulator to see how well the car was driving around track two. There were a few spots where the vehicle fell off the track specially at the sharp turns or very steep road.To improve the driving behavior in these cases, I recommend more data is needed to be collected

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. 


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover and return back to the center of the road. 

Then I repeated this process on track one clockwise and anticlockwise in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would give more data to the network. 
line :49-52 , 88-91 

I  randomly shuffled the data set it doesn't work. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. using video.py

python video.py Full_track_1 Creates a video based on images found in the Full_track_1 directory. The name of the video will be the name of the directory followed by '.mp4', so, in this case the video will be Full_track_1.mp4.

The default FPS is 60.

Please run Full_track_1.mp4
