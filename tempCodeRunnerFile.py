

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='mse'
)

early_stop = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(