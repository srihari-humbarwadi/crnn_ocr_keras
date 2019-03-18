from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, CuDNNLSTM, CuDNNGRU, Bidirectional, TimeDistributed, Reshape, Lambda, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, multi_gpu_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from glob import iglob
from tqdm import tqdm
import numpy as np
import pickle
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


ngpus = 4
batch_size = int(sys.argv[1])
epochs = int(sys.argv[2])

image_list = list(iglob('../synth/*'))
image_list.sort()
print(f'=> Found {len(image_list)} images <=')

w = []
h = []
max_w = 800
max_h = 32

# for img in tqdm(image_list):
#     w.append(img_to_array(load_img(img)).shape[1])
#     h.append(img_to_array(load_img(img)).shape[0])
# max_w = max(w) if max(w) % 2 == 0 else max(w) + 1
# # assert len(set(h)) == 1
# max_h = h[0]
print(f'=> Max width of images {max_w} <=')

labels = ' '.join([x.split('/')[-1].split('_')[1] for x in image_list])
vocab = sorted(list(set(labels)))
vocab_size = len(vocab)
print(f'=> Vocab size of dataset {vocab_size} <=')

letter_idx = {x: idx for idx, x in enumerate(vocab)}
idx_letter = {v: k for k, v in letter_idx.items()}

string_lens = [len(x) for x in [x.split('/')[-1].split('_')[1] for x in image_list]]
max_string_len = max(string_lens)
print(f'=> Max string len {max_string_len} <=')


h, w = max_w, max_h


def ctc_loss(tensor_list):
    y_pred, y_true, input_length, label_length = tensor_list
    y_pred = y_pred[:, 2:, :]
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)


def dummy_loss(y_true, y_pred):
    return y_pred


# print('=> Building model <=')
# input_layer = Input(shape=(h, w, 1), name='image_input')
# y = Conv2D(filters=32, kernel_size=3, padding='same',
#            kernel_initializer='he_normal', activation='relu', name='conv2d_1')(input_layer)
# y = MaxPool2D(pool_size=(2, 2), name='maxpooling2d_1')(y)
# y = Conv2D(filters=32, kernel_size=3, padding='same',
#            kernel_initializer='he_normal', activation='relu', name='conv2d_2')(y)
# y = MaxPool2D(pool_size=2, name='maxpooling2d_2')(y)
# y = Conv2D(filters=32, kernel_size=3, padding='same',
#            kernel_initializer='he_normal', activation='relu', name='conv2d_3')(y)
# # y = MaxPool2D(pool_size=2, name='maxpooling2d_3')(y)
# y = Reshape(target_shape=(h // 4, w // 4 * 32), name='reshape')(y)

# y = Bidirectional(CuDNNLSTM(units=512, return_sequences=True),
#                   name='biLSTM_1')(y)
# y = Bidirectional(CuDNNLSTM(units=512, return_sequences=True),
#                   name='biLSTM_2')(y)
# output_layer = TimeDistributed(Dense(
#     units=vocab_size+1, kernel_initializer='he_normal', activation='softmax'), name='char_output')(y)

# labels = Input(shape=(max_string_len, ))
# label_length = Input(shape=(1,))
# input_length = Input(shape=(1,))
# loss_layer = Lambda(ctc_loss, output_shape=(1,), name='loss_layer')(
#     [output_layer, labels, input_length, label_length])
# input_tensors = [input_layer, labels, label_length, input_length]
# train_model = Model(inputs=input_tensors, outputs=loss_layer)
# print('=> Build completed successfully <=')
# print(f'=> Creating model replicas for distributed training across {ngpus} gpus <=')

downscale_factor = 4
print('=> Building model <=')
base_model = ResNet50(weights=None, include_top=False, input_shape=(h,w,1))
conv_features = base_model.get_layer('activation_9').output
conv_features = Conv2D(filters=128, kernel_size=3, padding='same', kernel_initializer='he_normal')(conv_features)
y = Reshape(target_shape=(h // downscale_factor, w // downscale_factor * 128), name='reshape')(conv_features)
y = Bidirectional(CuDNNLSTM(units=512, return_sequences=True),
                  name='biLSTM_1')(y)
y = Bidirectional(CuDNNLSTM(units=512, return_sequences=True),
                  name='biLSTM_2')(y)
output_layer = TimeDistributed(Dense(
    units=vocab_size+1, kernel_initializer='he_normal', activation='softmax'), name='char_output')(y)

labels = Input(shape=(max_string_len, ))
label_length = Input(shape=(1,))
input_length = Input(shape=(1,))
loss_layer = Lambda(ctc_loss, output_shape=(1,), name='loss_layer')(
    [output_layer, labels, input_length, label_length])
input_tensors = [base_model.input, labels, label_length, input_length]
train_model = Model(inputs=input_tensors, outputs=loss_layer)
print(train_model.summary())
print('=> Build completed successfully <=')
print(f'=> Creating model replicas for distributed training across {ngpus} gpus <=')

pmodel = multi_gpu_model(train_model, ngpus)
pmodel.compile(optimizer=Adam(5e-5), loss=dummy_loss, metrics=['accuracy'])


class ocr_generator(Sequence):
    def __init__(self, images_list, img_h, img_w, batch_size, max_string_len=max_string_len, downscale_factor=downscale_factor):
        self.x_set = images_list
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_string_len = max_string_len
        self.downscale_factor = downscale_factor

    def __len__(self):
        return len(self.x_set) // batch_size + 1

    def __getitem__(self, idx):
        batch_start_idx = idx * self.batch_size
        batch_end_idx = (idx + 1) * self.batch_size
        assert batch_end_idx - batch_start_idx == self.batch_size
        batch_img = []
        batch_label = []
        batch_label_len = []
        batch_input_len = []

        batch_targets = []
        for img in self.x_set[batch_start_idx:batch_end_idx]:
            blank_img = np.ones((self.img_h, self.img_w))
            temp_img = img_to_array(load_img(img, color_mode='grayscale'))/255.
            blank_img[:temp_img.shape[0], :temp_img.shape[1]] = temp_img[:, :, 0]

            temp_label = [letter_idx[x] for x in img.split('/')[-1].split('_')[1]]
            temp_label_len = len(temp_label)
            x = temp_img.shape[1]
            y = self.downscale_factor
            temp_input_len = (x // y) - 2
            _img = np.expand_dims(blank_img.T, axis=-1)
            batch_img.append(_img)
            batch_label.append(pad_sequences(
                [temp_label], maxlen=self.max_string_len, padding='post', value=-1)[0])
            batch_label_len.append(temp_label_len)
            batch_input_len.append(temp_input_len)
            batch_targets.append(0)
        return (np.array(batch_img), np.array(batch_label), np.array(batch_label_len), np.array(batch_input_len)), np.array(batch_targets)


train_image_list = list(iglob('../dataset/train/*'))
val_image_list = list(iglob('../dataset/val/*'))
print(f'=> Found {len(train_image_list)} train images <=')
print(f'=> Found {len(val_image_list)} validation images <=')
print(f'=> Batch_size : {batch_size} <=')
print(f'=> Creating data generators <=')
train_generator = ocr_generator(train_image_list, max_h, max_w, batch_size)
val_generator = ocr_generator(val_image_list, max_h, max_w, batch_size)
train_steps = train_generator.__len__()
val_steps = val_generator.__len__()
print(f'=> Train steps per epoch : {train_steps} <=')
print(f'=> Val steps per epoch : {val_steps} <=')
tensorboard = TensorBoard(
    log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
callbacks_list = [ModelCheckpoint('top_weights.h5', monitor='val_loss', verbose=1, save_best_only=True),
                  tensorboard, EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
print('=> Created callback objects <=')

print(vocab)
print('=> Initializing training loop <=')
history = pmodel.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=epochs,
                               validation_data=val_generator, validation_steps=val_steps,
                               workers=8, use_multiprocessing=False, max_queue_size=500, callbacks=callbacks_list)
print('=> loading best weights <=')
pmodel.load_weights('top_weights.h5')
print('=> saving final model <=')
Model(inputs=input_layer, outputs=output_layer).save('inference_model.h5')