xView3: Dark Vessels Solution 
=============================

This is an experimental solution for the xView3 dark vessel detection
competition https://iuu.xview.us/ using SAR satellite image data.

I only investigated in a solution for a part of the stated problem. That is,
the detection of fishing vessels within the satellite images.

In particular, I'm interested in the comparision of the performance of
different deep neural network architectures on this problem.

Approach 1
---------

This solution is based on a UNet architecture https://arxiv.org/pdf/1505.04597.pdf. 
UNet's have been very succesfully applied to the task of
medical image segmentation. The idea is that their ability to detect small
changes in structure of medical images should also apply to very small objects
in satellite images.

The input channels are the VV, VH and the bathymetry tiff images. The output
image is comprised of a single channel representing the probability of whether
a given pixel is a fishing vessel or not. The bathymetry image is choosen as
an input so that the UNet can easily exclude pixels in the images that are
on land.

The training images are created as a Gaussian distribution centered around
the vessel location with a standard deviation proportional to the vessel
length, the proportion factor being a hyper-parameter of the model. If the
probabilities of two different vessels overlap, the maximum is choosen.

For the loss function, the weighted mean of the logit crossentropy between the
trained and image distribution for each pixel is taken. The weights are choosen
such that the far bigger occurence of pixels not containing any vessels is
balanced.

Here is a full training scene with generated image and the three used bands:

![scene](https://github.com/drsk0/xview/assets/827698/45e2358b-8a7d-49a3-9b7a-8024dc660f24)

Training
--------

The biggest challenge of this problem setting is the size of the data. A single
satellite image in the form of a tiff file can be several GB's. To be able to
train on a laptop with max 16GB of memory, the satellite images are tiled in
smaller images of 128x128 pixels. This is enough to easily contain a single
vessel. The tile size is another hyper-parameter of the model. 

The bathymetry images have a lower resolution than the VV and VH band images.
Therefore, the bathymetry are enhanced with a linear interpolation.

We only train the UNet with tiles containing at least one vessel and use a
batch size of 16 tiles.

We use the ADAM optimizer with default parameters.

Here is one of the generated tiles containing a vessel:

![tile](https://github.com/drsk0/xview/assets/827698/b8743414-416c-47bb-850b-9680ed09029c)

Optimizations
-------------

No optimzations have been tested so far. Batch size, tile size, number of
empty tiles within a batch, epoch size, the optimizer parameters etc. are prime
candidates for future optimizations.


Metric
------

To compare the different solutions, we use the f1_V metric as proposed in the
xview challenge.

Results
-------
A typical prediction of the model after 5 epochs of training:

![predict](https://github.com/drsk0/xview/assets/827698/f710e420-934d-412c-af08-ad736e0d4683)

And its corresponding ground truth: 

![predict_groundtruth](https://github.com/drsk0/xview/assets/827698/a01f29c6-fdad-4ef7-a625-197e06b26109)

(The plot implementations for Rasters and arrays have different orderings of one
of the axis)

The f1-score with a threshold of 0.25 on the validation dataset is 0.74.

A few insights:

  - A f1-score of 0.74 for the vanilla model is ok, but not impressive.
  - The model has high uncertainty for the boundaries of the tiles.
  - The threshold of 0.25 seems pretty low. The f1-score for higher thresholds, say 0.7, are very close however.
  - More data doesn't seem to improve the model significantly.


Approach 2
----------

It would be interesting to see whether a transformer outperforms the UNet on
this task.