checkpoint_callback = ModelCheckpoint(
    monitor="val_precision",
    mode="max",
    filename="maneuver-{epoch:02d}-{val_precision:.2f}",
    save_top_k=2
)

early_stop = EarlyStopping(
    monitor="val_precision",
    patience=10,
    mode="max",
    check_on_train_epoch_end=False
)
