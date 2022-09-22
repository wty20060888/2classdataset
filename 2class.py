import numpy as np
import glob
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# kaggle中修改
full_path = glob.glob("./*/*.jpg")
# full_path[:4]
label_to_index = {"lake":0, "airplane":1}
full_path[0].split("/")[1]
label = [i.split("/")[1] for i in full_path]
index_to_label = dict((v, k) for (k, v) in label_to_index.items())
all_index = [label_to_index.get(i) for i in label]

# 读取图片
img = full_path[100]
img_raw = tf.io.read_file(img)
## 解码
img_tensor = tf.image.decode_jpeg(img_raw)
img_tensor.shape
img_tensor.dtype

## 转化为float32类型，uint8不适合做计算
img_tensor = tf.cast(img_tensor, tf.float32)
img_tensor = img_tensor / 255
img_tensor.numpy().max()
img_tensor.numpy().min()

def load_img(path):
    img_raw = tf.io.read_file(path)
    img_tensor = tf.image.decode_jpeg(img_raw,channels=3)
    img_tensor = tf.image.resize(img_tensor,(256, 256))
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor / 255
    return img_tensor

i = random.choice(range(len(full_path)))
img = full_path[i]
imgx = load_img(img)
label = all_index[i]
plt.title(f"{index_to_label.get(label)}")
plt.imshow(imgx.numpy())
# %matplotlib auto

# 创建dataset
img_dataset = tf.data.Dataset.from_tensor_slices(full_path)
img_dataset = img_dataset.map(load_img)
img_dataset


label_dataset = tf.data.Dataset.from_tensor_slices(all_index)
dataset = tf.data.Dataset.zip((img_dataset, label_dataset))
dataset = dataset.shuffle(1400)

image_count = len(full_path)
test_count = int(image_count * 0.2)
train_count = image_count - test_count

train_dataset = dataset.skip(test_count)
test_dataset = dataset.take(test_count)

BATCH_SIZE = 16

#for i in dataset.take(10):
#    print(i[1])

train_dataset = train_dataset.repeat().shuffle(100).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# 创建模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3),input_shape=(256, 256, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(256, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu"))
model.add(tf.keras.layers.Conv2D(512, (3, 3), activation="relu"))
#model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024,activation="relu"))
model.add(tf.keras.layers.Dense(256,activation="relu"))
model.add(tf.keras.layers.Dense(1,activation="sigmoid"))


model.summary()


model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['acc']
        )
STEPS_PER_EPOCH = train_count // BATCH_SIZE
VAL_STEPS_PER_EPOCH = test_count // BATCH_SIZE
history = model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=test_dataset,
        validation_steps=VAL_STEPS_PER_EPOCH
        )


new_model = tf.keras.models.load_model("/Users/wangtianyu/Library/CloudStorage/OneDrive-个人/文档/日月光华-tensorflow入门与实战资料/预训练权重/2class.h5")
new_model = tf.keras.models.load_model("../../预训练权重/2class.h5")
model.load_weights("/Users/wangtianyu/Library/CloudStorage/OneDrive-个人/文档/日月光华-tensorflow入门与实战资料/预训练权重/2class_weights.h5")

model.evaluate(test_dataset)
test_img = load_img("/Users/wangtianyu/Library/CloudStorage/OneDrive-个人/文档/日月光华-tensorflow入门与实战资料/数据集/2_class/airplane/airplane_001.jpg")
test_img = tf.expand_dims(test_img, 0)
test_img.shape
index_to_label[0]
model.predict(test_img)

model.save_weights("testckpt")

model.load_weights("./testckpt")

checkpoint = tf.train.latest_checkpoint("./")
t = tf.train.Checkpoint()
t.restore("./testckpt")

model.save("./model/model.h5")

model = tf.keras.models.load_model("./model/model.h5")
