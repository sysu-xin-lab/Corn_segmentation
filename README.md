# Corn_segmentation

Compile:
--------

1. Compile the tf_sampling module of pointCNN following the guidelines in https://github.com/yangyanli/PointCNN
2. Unzip the pretrained pointCNN model in folder /pointCNN/save
3. Compile the c++ code in folders /cornExtract and /stemExtract, expectively. Note that PCL c++ library is required, which can be installed via https://pointclouds.org/downloads/. For windows users, we have provided executable files in the folder /data/cornExtract.exe and /data/stemExtract.exe. 

Reproduce the results:
--------

1. All data and codes are listed in the folder /data. "testdata_original.txt" is the test data in the form of XYZ.
2. Because the PointCNN only supports a specific number of points (2048) as the input data, we divided the testdata into grids, and normalized the points in each grid by running the following matlab code:  
    ```
    proc_1_normalizeTestdata.m
    ```
    Then we got 3 files, where the "testdata_nor.txt" is the normalized point cloud data, params.mat is the normalizing paramaters, and if a grid does not contain enough points, the points are recorded in the remain.mat 
3. run the 
