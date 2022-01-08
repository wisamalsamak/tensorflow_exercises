import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load(
    'cifar100',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache().shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128).prefetch(AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(128).prefetch(AUTOTUNE)

class CNNBlock(layers.Layer):
    def __init__(self, num_channels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(num_channels, kernel_size, padding='same')
        self.relu = tf.nn.relu
        
    def call(self, input_tensor):
        x = self.conv(input_tensor)
        return self.relu(x)
    
class ResidualBlock(layers.Layer):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.cnn1 = CNNBlock(channels[0], 1)
        self.cnn2 = CNNBlock(channels[1])
        self.cnn3 = CNNBlock(channels[2], 1)
        self.relu = tf.nn.relu
        # self.identity_mapping = layers.Conv2D(channels[1], 1, padding='same')
        self.identity_mapping = layers.Conv2D(channels[2], 1, padding='same')
    def call(self, input_tensor):
        x = self.cnn1(input_tensor)
        x = self.cnn2(x)
        # x = self.cnn3(x + self.identity_mapping(input_tensor))
        x = self.cnn3(x)
        return tf.nn.relu(x + self.identity_mapping(input_tensor))

class ResNet(keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.block1 = ResidualBlock([64, 64, 256])
        self.block2 = ResidualBlock([128, 128, 512])
        self.block3 = ResidualBlock([256, 256, 1024])
        self.block4 = ResidualBlock([512, 512, 2048])
        self.conv = layers.Conv2D(64, 7, padding='same')
        self.maxpool = layers.MaxPooling2D((3,3))
        self.averagepool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(128, activation=tf.nn.relu)
        self.classifier = layers.Dense(num_classes, activation=tf.nn.softmax)
        
    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.maxpool(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.averagepool(x)
        x = self.dense(x)
        return self.classifier(x)

model = ResNet(num_classes=100)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)
model.fit(ds_train, epochs=5, verbose=2)
model.evaluate(ds_test, verbose=2)
                
        