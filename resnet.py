# -*- coding: utf-8 -*-
import os, glob, time
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
from keras.constraints import unit_norm
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input

FC_NUMS = 64
batch_size = 10
EPOCHS=50

train_count = len(glob.glob('./PetImages/train/*/*.jpg'))
valid_count = len(glob.glob('./PetImages/test/*/*.jpg'))

train_datagen = image.ImageDataGenerator(
    rescale = 1./255, 
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
valid_datagen = image.ImageDataGenerator(
    rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        './PetImages/train/',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary')
validation_generator = valid_datagen.flow_from_directory(
        './PetImages/test',
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary')

def build_model(istrain):
    input = Input(shape = (256,256,3),name="kfb_image")
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input) 
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_NUMS, activation='relu', kernel_constraint=unit_norm())(x)
    if istrain:
        x = Dropout(0.2)(x)
    prediction = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=prediction)

    # opt = Adam(lr=INIT_LR, decay=1e-5)
    # model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    return model

def train(train_count,valid_count,istrain=False):
    model = build_model(istrain)
    models_save_path = "./models"
    if not os.path.exists(models_save_path):
        os.makedirs(models_save_path)

    checkpoint = ModelCheckpoint(filepath=os.path.join(models_save_path, 'resnet-{epoch:02d}-{val_acc:.4f}.h5'),
                                 monitor='val_acc',
                                 mode='max',
                                 save_best_only=True,
                                 save_weights_only=True)
    print("Train files: {}, valid files: {}".format(train_count,valid_count))
    print('-----------Start training-----------')
    start = time.time()
    model.fit_generator(train_generator,
                        steps_per_epoch=train_count // batch_size,
                        epochs=EPOCHS,
                        initial_epoch=0,
                        validation_data=validation_generator,
                        validation_steps=valid_count // batch_size,
                        callbacks=[checkpoint],
                        use_multiprocessing=False)
    end = time.time()
    print("train finished, cost time = {} hours".format(round((end - start) / 3600.0,3)))

def test(model_path,image_dir):
    model = build_model(False)
    model.load_weights(model_path, by_name=True)
    images = glob.glob(image_dir+"/*.jpg")
    preds=0
    for imgs in images:
        im = image.load_img(imgs, target_size=(256, 256))
        im = image.img_to_array(im)* 1. / 255
        im = np.expand_dims(im, axis=0)
        im =  preprocess_input(im)
        out = model.predict(im)
        print(np.argmax(out,axis=1))
        preds += out
    if preds/len(images)>0.5:
        print("{}: cat".format(os.path.basename(image_dir)))
    else:
        print("{}: dog".format(os.path.basename(image_dir)))


        


if __name__=="__main__":
    # train(train_count,valid_count,istrain=True)
    test('./models/resnet-03-0.6081.h5',"./PetImages/test/dog")

