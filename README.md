# Facial Keypoint detection

The start code is from Daniel Nouri's tutorial, but mine focused on the parameter turning and code format.

## Dataset
You can find the dataset [here](https://www.kaggle.com/c/facial-keypoints-detection/data). The training dataset for the Facial Keypoint Detection challenge consists of 7,049 96x96 gray-scale images. For each image, we're supposed learn to find the correct position (the x and y coordinates) of 15 keypoints, such as *left_eye_center*, *right_eye_outer_corner*.

Run my code of 'loader.py', you may find an interesting twist with the dataset is that for some of the keypoints we only have about 2,000 labels, while other keypoints have more than 7,000 labels available for training. So I drop all rows that have missing values in them, which has only 2,140 samples left.

Note: in practice, it is better to transform the point coordinate from [0,95]\*[0,95] to [-1,1]\*[-1,1]. Also, need to scale the pixel from 0~255 to 0~1, to get over the ill condition of the dataset. 

## Result

### 3 layers MLP structure (a glance)
The structure is as following:
```
  #  name      size
---  ------  ------
  0  input     9216
  1  hidden     100
  2  output      30
```

Before parameter tuning by Daniel Nouri's tutorial:
![figuresdsf](/figures/mlp_beforetuning.png)

After parameter tuning by myself:

![figuresdsf](/figures/mlp_aftertuning.png)

### CNN
The structure is as following:
```
  #  name      size
---  --------  ---------
  0  input     1x96x96
  1  conv1     32x94x94
  2  pool1     32x47x47
  3  dropout1  32x47x47
  4  conv2     64x46x46
  5  pool2     64x23x23
  6  dropout2  64x23x23
  7  conv3     128x22x22
  8  pool3     128x11x11
  9  dropout3  128x11x11
 10  hidden4   1000
 11  dropout4  1000
 12  hidden5   1000
 13  output    30
 ```

 Takes huge time to run!




