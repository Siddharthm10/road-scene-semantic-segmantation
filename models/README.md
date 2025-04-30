# Model Architectures:


## U-Net Architecture
#### Features:
1. Fully Convolutional Encoder + decoder architecture
2. Repeated 3x3 conv layers + ReLU + Max Pooling
3. Continuous Downsampling
4. Transposed convolutions for upsampling
5. Skip connections from parallel Encoder layer to Decoder layer

#### Results:
* Performs well on small datasets and fine boundary detection
* Lightweight and easy to train
* Retains fine grained and spatial features due to skip connections

## DeepLabV3+ Architecture
#### Features
1. Backbone can vary as per needs of the problem statement
2. ASPP (Atrious Spatial Pyramid Pooling) - captures multi-scale context
3. Contains image level pooling to gather global context
4. Refines segmentation outputs
5. Combines low level features from early layer with high level features from ASPP

#### Results:
* multi-scale context
* State-of-the-art accuracy
* Better generalization


### What is Dilated convolutions?
Here we add skip pixels between adjacent pixel connections. Thus allowing the CNN filter to have a higher receptive field without actually increasing the number of parameters, or loosing feature map resolutions. 
```
Dilation 1 (Normal):
[ p1 p2 p3 ]
[ p4 p5 p6 ]
[ p7 p8 p9 ]

Dilation 2 (Atrous):
[ p1  -  p2  -  p3 ]
[ -   -  -   -  -  ]
[ p4  -  p5  -  p6 ]
[ -   -  -   -  -  ]
[ p7  -  p8  -  p9 ]
```

* Helps understand large contexts
* High accuracy
* Avoid degrading resolution



### What is ASPP?
* Run Dilated convolutions with different dilation rates, on the same feature map. 
* Plus an global average pooling branch
* Network can see the same object at different scales simultaneously.


