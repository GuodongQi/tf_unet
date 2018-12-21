UNET
-
we use UNET to predict the label of image. The main idea is : 

end to end ConvNet, make a grid of size 5 * 26 , logistics regression to predict whether 0/1

Dependence：
-
- python>3 
- opencv
- tensorflow
- matplotlib

Train
-
make your train.txt. Per line should be like:
```
path/to/image1 lanels1_to_string
path/to/image2 lanels2_to_string
path/to/image2 lanels3_to_string
```
make sure your path, then

`python train.py`

 During training , we can use TensorBoard to watch the training trace.

`tensorboard --logdir=path/to/logs --host=127.0.0.1`

Predict
-
`python unet.py`

make sure your image path and weights path