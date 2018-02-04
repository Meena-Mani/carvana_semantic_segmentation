### Deep learning binary semantic segmentation with U-Net and LinkNet

#### U-Net 
Here I compare a modified U-Net (which uses ResNet residual modules) with a modified U-Net that used dilated convolutions. The *U-Net+dilated convolution blocks* network won the [third place]() in the [Kaggle Carvana image segmentation competition]().  
The Jupyter Notebook and associated model files are in the *modified_unet* directory. 
 
#### LinkNet     
This is my implementation of LinkNet in Keras. I use this network on the Carvana Image Masking challenge dataset.  
The Jupyter Notebook and associated model files are in the *linknet* directory. 


#### Libraries   
These notebooks were developed with the following libraries/versions:  
- Python: 3.6.3 
- Keras:  2.1.2
- OpenCV
