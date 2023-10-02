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

to add new images to the datatset can be done by visiting the path `/app/AddTrainImage` The page is divided under 2 parts first is for adding one or more images to a particular class while the other is to test out results and get predictions on given images.

All the newly added images are stored in `User_Custom_Train` folder under the appropriate folder depicting the class name.