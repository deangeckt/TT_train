import pandas as pd
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

SEQ_LEN = 50
# INPUT_SIZE = 3 * 25  # x,y,z,vis * 25 points
INPUT_SIZE = 3 * (25-11)
head = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
str_head = [str(h) for h in head]


# y: ndarray (1200,)
# x: ndarray (1200, 50, 100)
def read_raw_data():
    score_df = pd.read_excel('labels/fco_score.xlsx', engine="openpyxl")
    score_df = score_df[score_df['frames'] <= SEQ_LEN]

    data_df = pd.read_csv('labels/fco_data.csv')
    amount_of_shots = 5 #len(score_df)
    print('all examples:', amount_of_shots)

    single_shot_x = np.zeros((SEQ_LEN + 1, INPUT_SIZE))
    x = np.zeros((amount_of_shots, SEQ_LEN + 1, INPUT_SIZE))
    y = []
    shot_index = 0

    for _, row in tqdm(score_df.iterrows()):
        frames = row['frames']
        score = row['score']
        name = row['name']

        for i in range(0, frames):
            frame_name = name + '_{}'.format(i)
            shot_data = []
            for k, v in data_df[data_df['name'] == frame_name].iteritems():
                # optional feature - TODO - should be controlled outside
                if k == 'name' or k.endswith('vis'):
                    continue
                if any(k.split('_')[0] == h for h in str_head):
                    continue

                shot_data.append(v.values[0])
            single_shot_x[i] = np.array(shot_data)

        if shot_index == amount_of_shots:
            break

        y.append(score)
        x[shot_index] = np.nan_to_num(single_shot_x)
        shot_index += 1

    return x, np.array(y)


x, y = read_raw_data()
print('x shape', x.shape)
print('y shape', y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print('x train', x_train.shape)


model = keras.models.Sequential()
# model.add(keras.Input(shape=(SEQ_LEN + 1, INPUT_SIZE)))
# model.add(layers.SimpleRNN(64, return_sequences=True, activation='relu'))  # hidden cells
# model.add(layers.SimpleRNN(256, return_sequences=False, activation='relu'))
model.add(layers.LSTM(16, activation='relu', input_shape=(SEQ_LEN + 1, INPUT_SIZE)))
model.add(layers.Dense(1))


optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = tf.keras.metrics.MeanSquaredError()
loss = tf.keras.losses.MeanAbsoluteError()

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 10
model.fit(x, y, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

# model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

# prob_model = keras.models.Sequential([model, keras.layers.Softmax()])
# pred0 = prob_model.predict(x_test)[0]
# label0 = np.argmax(pred0)
# print(label0)