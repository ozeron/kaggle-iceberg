from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

def build_save_callbacks(filepath, patience=2):
#     es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
#     return [es, msave]
    return [msave]
