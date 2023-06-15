from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
datagen = ImageDataGenerator(
        rotation_range=1.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
img = load_img('Driveway.jpg')
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='ImageDataSet/generatedImages', save_prefix='cctv', save_format='jpeg'):
    i += 1
    if i > 5:
        break  # otherwise the generator would loop indefinitely