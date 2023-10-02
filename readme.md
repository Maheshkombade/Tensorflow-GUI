# 1 Installation and user manual
This document gives easy steps to install/set-up and to use the web app which will enable to user to interact with the model.

## 1.1 Installation and Setup
>`Note` The user should have python 3.7 installed. <br>
Go the drive link [link](https://drive.google.com/drive/folders/1cJDIbhzLYxPJarkTMnOBG4ssflOfFHw4?usp=sharing)

Extract the files from the INTER_IIT folder,there would be one folder:

- `BOSCH_sign` is the folder having all the `code+ training data + models + requirements.txt`.

## Create the virtual environment:
Create a virtual envirionmnent in the choice of your directory by running the following command in the `terminal`
```
$python3.7 -m venv env-name
```
Activate the virtual environment by running the following command.
```
source env-name/bin/activate
```
After that, install all the required dependencies from the `requirements.txt` file that is stored in the `BOSCH_sign` folder
```
pip install -r requirements.txt
```







### 1.1.3 Run the Django server.

- cd into the BOSCH_sign folder and run the following commands to start the server. <br>
`NOTE` In case of a crash rerun this command
```
python3.7 manage.py runserver

```
- make migrations, the make migrations will only be needed to be done once.
```
$ python3.7 manage.py makemigrations

$ python3.7 manage.py migrate
```


## 1.2 Usage
After carefully going over the problem statement the functionalty have been divided onto broadly four types/coditions.

1. The user should be able to enter images in the training set which consists of 42 default classes and 5 additional classes added by us.

2. The user should be able to retrain the model after making augmentations to the images in the newly added 5 classes and then merging the augmented image with the existing dataset to increase the difficulty OR  apply augmentations to the whole training set and then merge it with the exixting trainign set in both the cases the difficulty and the scale of the training data would be increased and see the statistics of the model.
3. Test the model for any image.
4. Make neuron level analysis so that the reason for the models performance can be found.
5. The user should be able to make augmentations to just the original 43 classes and then add the augmented images to the existing training set thereby increaseing the data set and its difficulty.
6. Display and compare the results of retaining with the initial original trained model and the new one through varoius graphs.

To solve the above cases diffrent pathways(url paths/links) have been made with diffrent functionalities.Below is the the page level explanation of the functionalities.

### 1.1.2 Homepage.



#### Navbar
Everypage will have the navbar and The navbar has two buttons:<br>
1. Home:<br>
This button will bring you back to the home page from anywhere
2. Add images:<br>
This will take you to "/app/AddTrainImage/" . The form will will enable the user to entern n number of images to any one of the 48 classes.
3. Test Images:<br>
This will take you to "/app/TestImage/" . The form will will enable the user to enter a image and predict its class.


#### Localhost:8000/app/home:

> :warning: *Whereever you see an opton to retrain press it only once and wait for some time as it might take a bit to train the model clicking a retrain button multiple time will initiate multiple retrains in parallel you can check the training process on your terminal while retraining*!

The page has 4 pipelines all adressing one of the above problems.<br>
- The first button will simply retrain the original model of 43 classes and show that the default graphs that are displayed later in the webpage will match the new graphs that would be generated.<br>
`button1-->retraining-->graphs-->XAI analysis`

- The second button will take you to the augmenttation page and enable you to add augmentations to the original training data then the user will be displayed one randomply chosen pair or original vs augmented from each class.
After verifiying that the augmentations are corrctly done the user can then retrain the data after which he will be displayed graphs of the model before and after augmentations.<br>
`button2-->augmentation-->original vs augmented images-->retraining-->graphs-->XAI Analysis`

- The third button will take you to the add images page where the user can add images to the 48 classes and then he would be directed to an intermediate page (by clicking the `retrain button`
at the end of the form)
 that would check if the user wants to apply the augmentations to the newly added 5  classes or to the whole data set.After which the user would be directed to a webpage where he can see the original vs augmented image from each augmented class.The augmeted images would then be added with the training data and the model would be trained again.The page will then be directed to graphs webpage where he can see the graphs showing the statistics of the original vs the augmented model .<br>
`button3--> augment and then merege or merge and then augment--> augmentation--> original vs augmented images--> retraining--> graphs--> XAI analysis`

- The fourth button will take the user to the testTraining image page where the user would be able to test any image and predict its class. The images class would be predicted using the default model that was trained over the given 43 plus the additional 5 classes that were added by us.

- The fifth button will show you the graphs of the model.

>`Note`: We have used session variables to check which case is in working right now hence making migrationas as specified above is a must and the user should most ideadly follow the pipeline going straight to a particular `url`that lies in the middle of the pipeline should be avoided.

>At the end of each case the user would be on `app/graphs` page where if the user wants to see what the model sees he can click the XAI analysis button to do the analysis.

>All the Models would be saved in the `classification/model` directory




# Augmentations

The augmentations are added to image by visiting the `/app/augment` relative path on the server for the platform.

The LHS of the augmentations are the available types of augmentations that can be applied on the images. On clicking a augmentation it will turn green and get selected on the RHS also. In case you want to remove this augmentatoin from the list you can simply reclick on that augmentation from the LHS.

Once the augmentations are selected click on `submit` once all augmentations have been added

You will be redirected to a sample viewing page. which will show how the augmentatitons will alter the image by picking up  a random image from each class and show the effect of the augmentations on it.

### Note

The augmentations depend on the order in which they are selected. that is,

```
zoom
rotate
```

is not the same as

```
rotate
zoom
```

# Add Training data

to add new images to the datatset can be done by visiting the path `/app/AddTrainImage` .

All the newly added images are stored in `User_Custom_Train` folder under the appropriate folder depicting the class name.

# Graphs
In `home/app/graphs` the graphs corresponding to the original trained model and the graphs obtained after adding augmented images to the model are displayed. They include plots of accuracy and loss of the model along with the confusion matrix obtained after retraining it. The first three plots correspond to  the original model and are always displayed on the page and the remaining graphs are displayed when the model is run on any of the given cases.

## Inferences from the Graphs:
From the first graph “Accuracy”, we can infer the accuracy of the model corresponding to the number of epochs. There are two lines : the blue line depicts the training accuracy and the red line depicts the value accuracy of the model. Similarly, from the second graph “Loss”, we can infer the loss of the model corresponding to the number of epochs. There are two lines : the blue line depicts the training loss and the red line depicts the value loss of the model.  The final plot is a heatmap of the confusion matrix.

## Functionalities of the graphs:
When hovered upon any plot, a toolbar consisting of all available options appears on right side just above the plot. It provides the user tools for analyzing data and making graphs which are listed below in a detailed manner.
More information about a data point is revealed by moving your mouse cursor over the point and having a hover label appear which indicates the X and Y values corresponding to that data point.
If the plot's drag mode is set to 'Zoom', click and drag on the plot to zoom-in and double-click to zoom-out completely, i.e., autoscale both the axes.


To zoom along only one axis, click and drag near the edges of either one of the axes. Additionally, to zoom-in along both the axes together, click and drag near the corners of both the axes.

If the plot's drag mode is set to 'Pan', click and drag on the plot to pan and double-click to reset the pan


.
To Pan along one axis, click and drag from the center of the axis.

Double-click on a single axis to autoscale along that axis alone.


## XAI Analysis
 After the user has analysed the graphs displayed on the `app/graphs` page he can have the app do an XAI analysis on the model after which he would be redirected to a webpage where the app would give some insights as to why is model is behaving in such a way also there would be two images displayed that would show the user what the model sees in the image to clasfy them before and after augmentation.




