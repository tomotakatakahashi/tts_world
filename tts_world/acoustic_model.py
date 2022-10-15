
def get_model(input_dim = 329, f0_dim=1, sp_dim=513, ap_dim=513):
    ipt = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256,)(ipt)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(128,)(x)
    x = tf.keras.layers.ReLU()(x)
    sp_res = tf.keras.layers.Reshape((1, -1))(tf.keras.layers.Dense(32, activation='relu')(x))
    sp_space = tf.keras.layers.Reshape((-1, 1))(tf.keras.layers.Dense(16, activation='relu')(x))
    ap_res = tf.keras.layers.Reshape((1, -1))(tf.keras.layers.Dense(32, activation='relu')(x))
    x = tf.keras.layers.Dense(64,)(x)
    x = tf.keras.layers.ReLU()(x)

    f0_out = tf.keras.layers.Dense(1, name="f0")(x)

    x = tf.keras.layers.Reshape((1, -1))(x)

    sp_mid = tf.keras.layers.Conv1DTranspose(64, 7, strides=4, padding='same')(x)
    sp_mid = tf.keras.layers.ReLU()(sp_mid)
    sp_mid = tf.keras.layers.Conv1DTranspose(32, 7, strides=4, padding='same')(sp_mid)
    sp_mid = tf.keras.layers.Add(name='sp_add')([sp_mid, sp_res, sp_space])
    sp_mid = tf.keras.layers.ReLU()(sp_mid)
    sp_mid = tf.keras.layers.Conv1DTranspose(16, 7, strides=4, padding='same')(sp_mid)
    sp_mid = tf.keras.layers.ReLU()(sp_mid)
    sp_mid = tf.keras.layers.Conv1DTranspose(8, 7, strides=4, padding='same')(sp_mid)
    sp_mid = tf.keras.layers.ReLU()(sp_mid)
    sp_mid = tf.keras.layers.Conv1DTranspose(4, 7, strides=2, padding='same')(sp_mid)
    sp_mid = tf.keras.layers.Conv1DTranspose(1, 2, strides=1, padding='valid')(sp_mid)
    sp_out = tf.keras.layers.Flatten(name="sp")(sp_mid)

    ap_mid = tf.keras.layers.Conv1DTranspose(64, 7, strides=4, padding='same')(x)
    ap_mid = tf.keras.layers.ReLU()(ap_mid)
    ap_mid = tf.keras.layers.Conv1DTranspose(32, 7, strides=4, padding='same')(ap_mid)
    ap_mid = tf.keras.layers.Add()([ap_mid, ap_res])
    ap_mid = tf.keras.layers.ReLU()(ap_mid)
    ap_mid = tf.keras.layers.Conv1DTranspose(16, 7, strides=4, padding='same')(ap_mid)
    ap_mid = tf.keras.layers.ReLU()(ap_mid)
    ap_mid = tf.keras.layers.Conv1DTranspose(8, 7, strides=4, padding='same')(ap_mid)
    ap_mid = tf.keras.layers.ReLU()(ap_mid)
    ap_mid = tf.keras.layers.Conv1DTranspose(4, 7, strides=2, padding='same')(ap_mid)
    ap_mid = tf.keras.layers.Conv1DTranspose(1, 2, strides=1, padding='valid')(ap_mid)
    ap_out = tf.keras.layers.Flatten(name='ap')(ap_mid)

    model = tf.keras.models.Model(inputs=ipt, outputs=(f0_out, sp_out, ap_out))
    return model

model = get_model()
model.summary()
