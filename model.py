import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


'''
These were the best parameters found with optuna parameter space study
We by default use this
'''
best_params = {
    'cnn_filters': 128,
    'cnn_kernel_size': 3,
    'lstm_units': 224,
    'dropout_rate': 0.31000000000000005,
    'learning_rate': 0.0011522903935867418
}

def create_model(cnn_filters, cnn_kernel_size, lstm_units, dropout_rate, learning_rate, time_steps, features):
    '''
    Contructs a CNN with a feature head that plugs into a Bi-directional LSTM. The x variable
    pertains to the concatenation of the model architecture
    '''

    inp = Input(shape=(time_steps, features))
    x = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu', padding='same')(inp)
    x = MaxPooling1D(pool_size=2)(x)
    #x = Conv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, activation='relu', padding='same')(inp)     # this model architecture was tried as well
    #x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), #use adam
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Currently not used, since best parameters were already found.
def objective(trial, X_train, y_train, X_val, y_val, max_len, combined_dim):
    '''
    Oputna Study Function:
    constructs and optuna study that searches for the best parameters
    Stops poor convergences early to save computation time.
    '''

    # best estimated parameter space. Reduced from a larger parameter space,
    #cnn_filters = trial.suggest_int('cnn_filters', 16, 256, step=32)
    #cnn_kernel_size = trial.suggest_categorical('cnn_kernel_size', [2, 10])
    #lstm_units = trial.suggest_int('lstm_units', 16, 256, step=32)
    #dropout_rate = trial.suggest_float('dropout_rate', 0.001, 0.5, step=0.1)
    #learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    cnn_filters = trial.suggest_int('cnn_filters', 32, 128, step=32)
    cnn_kernel_size = trial.suggest_categorical('cnn_kernel_size', [3, 5])
    lstm_units = trial.suggest_int('lstm_units', 32, 256, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.01, 0.41, step=0.1)
    learning_rate = trial.suggest_float('learning_rate', 5e-4, 1e-1, log=True)
    model = create_model(cnn_filters=cnn_filters,
                         cnn_kernel_size=cnn_kernel_size,
                         lstm_units=lstm_units,
                         dropout_rate=dropout_rate,
                         learning_rate=learning_rate,
                         time_steps=max_len,
                         features=combined_dim)
    
    #add early stopping for poor convergence, and restore previous best generalizing weights
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=5, # low epochs for parameter searching
              batch_size=32, 
              callbacks=[early_stop],
              verbose=0)
    _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    return val_accuracy

def train_final_model(X_train, X_val, y_train, y_val, max_len, combined_dim):
    '''
    Uses best params to initialize all of the hyperparameters.
    Constructs the final model and runs 13 epochs. Used to evaluate the test set.
    Follows Optuna function in construction.
    '''
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    model = create_model(
        cnn_filters=best_params['cnn_filters'],
        cnn_kernel_size=best_params['cnn_kernel_size'],
        lstm_units=best_params['lstm_units'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        time_steps=max_len,
        features=combined_dim
    )
    early_stop_final = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    model.fit(X_train_full, y_train_full, 
              epochs=13, 
              batch_size=32, 
              validation_split=0.1, 
              callbacks=[early_stop_final],
              verbose=1)
    return model

def evaluate_model(model, X_test, y_test):
    '''
    Constructs Classification Report for the model
    '''
    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("\nTest Accuracy: {:.4f}".format(test_accuracy))
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
