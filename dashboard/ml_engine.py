import os
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from django.conf import settings


@tf.keras.utils.register_keras_serializable()
class FeatureWiseAttention(Layer):
    """Custom attention layer used in the trained LSTM model.
    
    Receives [lstm_output (batch, T, H), raw_input (batch, T, F)].
    Computes attention weights from lstm_output, applies to raw_input.
    Output shape: (batch, T, F) — same as raw_input.
    """
    def __init__(self, n_features=None, **kwargs):
        super(FeatureWiseAttention, self).__init__(**kwargs)
        self.n_features = n_features

    def build(self, input_shape):
        if isinstance(input_shape, list):
            lstm_shape = input_shape[0]
        else:
            lstm_shape = input_shape
            
        n_feat = self.n_features if self.n_features is not None else 11
        self.W = self.add_weight(name='attention_weight',
                                 shape=(lstm_shape[-1], n_feat),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(n_feat,),
                                 initializer='zeros',
                                 trainable=True)
        super(FeatureWiseAttention, self).build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            lstm_out = inputs[0]   # (batch, T, H)
            raw_input = inputs[1]  # (batch, T, F)
        else:
            lstm_out = inputs
            raw_input = inputs

        e = tf.keras.backend.tanh(tf.keras.backend.dot(lstm_out, self.W) + self.b)
        alpha = tf.keras.backend.softmax(e) # (batch, T, F)

        context = raw_input * alpha  # (batch, T, F)
        return context

    def get_config(self):
        config = super(FeatureWiseAttention, self).get_config()
        config['n_features'] = self.n_features
        return config


class MLEngine:
    """Singleton ML engine that loads the trained LSTM model and associated artifacts."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLEngine, cls).__new__(cls)
            cls._instance.is_loaded = False
            cls._instance.model = None
            cls._instance.scaler_all = None
            cls._instance.scaler_target = None
            cls._instance.metadata = None
            cls._instance.feature_cols = None
            cls._instance.all_cols = None
            cls._instance.attention_weights = None
            cls._instance.seed_history = None
            cls._instance.load_model_artifacts()
        return cls._instance

    def load_model_artifacts(self):
        model_dir = os.path.join(settings.BASE_DIR, 'models')
        try:
            if not os.path.exists(model_dir):
                print(f"Warning: Model directory {model_dir} not found.")
                return

            # Load metadata first — it defines column names
            metadata_path = os.path.join(model_dir, 'training_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {
                    'look_back': 90,
                    'threshold': 88.3794,
                    'feature_cols': [
                        'curah_hujan_mm', 'cuaca_kode', 'smd_kanan_q_ls',
                        'smd_kiri_q_ls', 'tma_lag1', 'tma_lag2', 'tma_lag3',
                        'delta_tma', 'tma_rolling_mean_3', 'jam_kode'
                    ],
                    'all_cols': [
                        'tma_m', 'curah_hujan_mm', 'cuaca_kode', 'smd_kanan_q_ls',
                        'smd_kiri_q_ls', 'tma_lag1', 'tma_lag2', 'tma_lag3',
                        'delta_tma', 'tma_rolling_mean_3', 'jam_kode'
                    ],
                    'target_col': 'tma_m'
                }

            self.feature_cols = self.metadata.get('feature_cols', [])
            self.all_cols = self.metadata.get('all_cols', [])

            # Load model — handle Keras 2/3 compatibility
            model_path = os.path.join(model_dir, 'best_model.keras')
            if os.path.exists(model_path):
                self.model = self._load_keras_model(model_path)
                if self.model:
                    print(f"Model loaded: input shape {self.model.input_shape}")
            else:
                print(f"Warning: {model_path} not found.")

            # Load scalers
            scaler_all_path = os.path.join(model_dir, 'scaler_all.pkl')
            if os.path.exists(scaler_all_path):
                with open(scaler_all_path, 'rb') as f:
                    self.scaler_all = pickle.load(f)

            scaler_target_path = os.path.join(model_dir, 'scaler_target.pkl')
            if os.path.exists(scaler_target_path):
                with open(scaler_target_path, 'rb') as f:
                    self.scaler_target = pickle.load(f)

            # Load attention weights
            attention_weights_path = os.path.join(model_dir, 'attention_weights.npy')
            if os.path.exists(attention_weights_path):
                self.attention_weights = np.load(attention_weights_path)
            else:
                self.attention_weights = np.ones(len(self.all_cols)) / len(self.all_cols)

            # Load seed history (pre-scaled sliding window for cold start)
            seed_path = os.path.join(model_dir, 'seed_history.npy')
            if os.path.exists(seed_path):
                self.seed_history = np.load(seed_path)
                print(f"Seed history loaded: shape {self.seed_history.shape}")
            else:
                self.seed_history = None

            self.is_loaded = True
            print("ML Engine loaded successfully.")
        except Exception as e:
            print(f"Failed to load ML artifacts: {e}")
            import traceback
            traceback.print_exc()

    def _load_keras_model(self, model_path):
        """
        Load a .keras model, reconstructing architecture manually if needed
        to work around Keras 2→3 deserialization issues.
        """
        import zipfile
        import shutil
        from keras.layers import Input, LSTM, Dropout, Dense

        try:
            return load_model(
                model_path,
                custom_objects={'FeatureWiseAttention': FeatureWiseAttention},
                compile=False, safe_mode=False,
            )
        except Exception as e1:
            print(f"Direct load failed ({e1}), reconstructing architecture...")

        try:
            # Read config from the .keras ZIP to get layer details
            with zipfile.ZipFile(model_path, 'r') as z:
                config = json.loads(z.read('config.json'))

            layers_config = config['config']['layers']
            layer_map = {}
            for lc in layers_config:
                layer_map[lc['name']] = lc

            # Reconstruct the architecture based on the saved config
            # Input: (90, 11)
            look_back = self.get_look_back()
            n_features = len(self.all_cols)
            inp = Input(shape=(look_back, n_features), name='input_sequence')

            # LSTM 1: returns sequences, 128 units
            lstm1_cfg = layer_map['lstm_1']['config']
            x = LSTM(lstm1_cfg.get('units', 128),
                     return_sequences=True, name='lstm_1')(inp)
            x = Dropout(layer_map['dropout_1']['config'].get('rate', 0.2),
                        name='dropout_1')(x)

            # FeatureWiseAttention: takes [lstm_out, raw_input]
            attn = FeatureWiseAttention(n_features=n_features,
                                        name='feature_attention')([x, inp])

            # LSTM 2: 64 units, takes attention output, returns sequences
            lstm2_cfg = layer_map['lstm_2']['config']
            x2 = LSTM(lstm2_cfg.get('units', 64),
                      return_sequences=True, name='lstm_2')(attn)
            x2 = Dropout(layer_map['dropout_2']['config'].get('rate', 0.2),
                         name='dropout_2')(x2)

            # LSTM 3: 32 units, returns single vector
            lstm3_cfg = layer_map['lstm_3']['config']
            x3 = LSTM(lstm3_cfg.get('units', 32),
                      return_sequences=False, name='lstm_3')(x2)
            x3 = Dropout(layer_map['dropout_3']['config'].get('rate', 0.2),
                         name='dropout_3')(x3)

            # Dense layers
            x3 = Dropout(layer_map.get('dropout_pre_dense', {}).get('config', {}).get('rate', 0.2),
                         name='dropout_pre_dense')(x3)
            d1_cfg = layer_map['dense_1']['config']
            x3 = Dense(d1_cfg.get('units', 32),
                       activation=d1_cfg.get('activation', 'relu'),
                       name='dense_1')(x3)
            d2_cfg = layer_map['dense_2']['config']
            x3 = Dense(d2_cfg.get('units', 32),
                       activation=d2_cfg.get('activation', 'relu'),
                       name='dense_2')(x3)

            out_cfg = layer_map['output']['config']
            output = Dense(out_cfg.get('units', 1),
                           activation=out_cfg.get('activation', 'linear'),
                           name='output')(x3)

            model = tf.keras.Model(inputs=inp, outputs=[output, attn],
                                   name='LSTM_FeatureAttention')

            # Load weights from the .keras archive
            temp_dir = os.path.join(os.path.dirname(model_path), '_temp_weights')
            with zipfile.ZipFile(model_path, 'r') as z:
                z.extractall(temp_dir)

            # Weights are stored in model.weights.h5
            weights_path = os.path.join(temp_dir, 'model.weights.h5')
            if os.path.exists(weights_path):
                import h5py
                with h5py.File(weights_path, 'r') as f:
                    layers_group = f['layers']
                    
                    def set_layer_w(model_layer, h5_layer_name, is_lstm=False):
                        if is_lstm:
                            vars_grp = layers_group[h5_layer_name]['cell']['vars']
                            model.get_layer(model_layer).set_weights([
                                vars_grp['0'][()], vars_grp['1'][()], vars_grp['2'][()]
                            ])
                        else:
                            vars_grp = layers_group[h5_layer_name]['vars']
                            model.get_layer(model_layer).set_weights([
                                vars_grp['0'][()], vars_grp['1'][()]
                            ])

                    # Map Keras 2 H5 names to Keras 3 Model names
                    set_layer_w('lstm_1', 'lstm', is_lstm=True)
                    set_layer_w('lstm_2', 'lstm_1', is_lstm=True)
                    set_layer_w('lstm_3', 'lstm_2', is_lstm=True)
                    set_layer_w('feature_attention', 'feature_wise_attention', is_lstm=False)
                    set_layer_w('dense_1', 'dense', is_lstm=False)
                    set_layer_w('dense_2', 'dense_1', is_lstm=False)
                    set_layer_w('output', 'dense_2', is_lstm=False)
                    
                print("Weights loaded successfully via manual H5 mapping.")
            else:
                print("Warning: model.weights.h5 not found in archive.")

            shutil.rmtree(temp_dir, ignore_errors=True)
            return model

        except Exception as e2:
            print(f"Architecture reconstruction failed: {e2}")
            import traceback
            traceback.print_exc()
            return None

    def get_look_back(self):
        if self.metadata and 'look_back' in self.metadata:
            return self.metadata['look_back']
        return 90

    def get_threshold(self):
        if self.metadata and 'threshold' in self.metadata:
            return self.metadata['threshold']
        return 88.3794

    def get_model_metrics(self):
        """Return model performance metrics from metadata."""
        if self.metadata and 'metrics' in self.metadata:
            return self.metadata['metrics']
        return {}

    def get_model_info(self):
        """Return a dict of model info for the Model Info page."""
        return {
            'look_back': self.get_look_back(),
            'threshold': self.get_threshold(),
            'n_features': self.metadata.get('n_features', len(self.all_cols)) if self.metadata else len(self.all_cols),
            'target_col': self.metadata.get('target_col', 'tma_m') if self.metadata else 'tma_m',
            'trained_at': self.metadata.get('trained_at', 'N/A') if self.metadata else 'N/A',
            'epochs_trained': self.metadata.get('epochs_trained', 'N/A') if self.metadata else 'N/A',
            'dataset_rows': self.metadata.get('dataset_rows', 'N/A') if self.metadata else 'N/A',
            'batch_size': self.metadata.get('batch_size', 'N/A') if self.metadata else 'N/A',
            'train_end_date': self.metadata.get('train_end_date', 'N/A') if self.metadata else 'N/A',
            'val_end_date': self.metadata.get('val_end_date', 'N/A') if self.metadata else 'N/A',
        }

    def _build_all_cols_row(self, feature_dict, tma_value=0.0):
        """Build a single row with all_cols order (tma_m first, then features)."""
        row = [tma_value]  # tma_m placeholder
        for col in self.feature_cols:
            row.append(float(feature_dict.get(col, 0.0)))
        return row

    def predict_single(self, feature_dict):
        """
        Predict TMA from a single set of input features.
        
        feature_dict should contain keys matching self.feature_cols.
        Uses seed_history or database history for the sliding window.
        """
        threshold = self.get_threshold()

        if not self.is_loaded or self.model is None:
            return np.random.uniform(85, 90), "Normal", threshold

        try:
            look_back = self.get_look_back()

            # Build the new row in all_cols order (tma_m = 0 placeholder)
            new_row_values = self._build_all_cols_row(feature_dict)

            # Scale this new row
            new_row_scaled = self.scaler_all.transform([new_row_values])[0]

            # Build the sliding window
            if self.seed_history is not None:
                # Use seed history as base, append new row, take last look_back
                window = np.vstack([self.seed_history, new_row_scaled.reshape(1, -1)])
                window = window[-look_back:]
            else:
                # Repeat the new row look_back times (poor fallback)
                window = np.tile(new_row_scaled, (look_back, 1))

            # Reshape for LSTM: (1, look_back, n_features)
            X = np.expand_dims(window, axis=0)

            # Predict
            preds = self.model.predict(X, verbose=0)
            pred_scaled = preds[0] if isinstance(preds, list) else preds

            # Inverse transform the target
            pred_value = self.scaler_target.inverse_transform(pred_scaled)[0][0]

            status = "Bahaya" if pred_value >= threshold else "Normal"
            return float(pred_value), status, threshold

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, "Error", threshold

    def predict_batch(self, df):
        """
        Predict TMA for a batch DataFrame.
        
        The DataFrame must contain columns matching self.feature_cols.
        Returns the DataFrame with added 'tma_predicted' and 'status' columns.
        """
        threshold = self.get_threshold()

        if not self.is_loaded or self.model is None:
            df['tma_predicted'] = np.random.uniform(85, 90, size=len(df))
            df['status'] = df['tma_predicted'].apply(
                lambda x: "Bahaya" if x >= threshold else "Normal"
            )
            return df

        try:
            look_back = self.get_look_back()

            # Build all_cols DataFrame: prepend tma_m column (zeros as placeholder)
            all_data = pd.DataFrame()
            all_data['tma_m'] = 0.0  # placeholder
            for col in self.feature_cols:
                if col in df.columns:
                    all_data[col] = df[col].values
                else:
                    all_data[col] = 0.0

            # Scale with scaler_all (expects all_cols order)
            data_scaled = self.scaler_all.transform(all_data[self.all_cols].values)

            # Prepend seed history if available for initial window
            if self.seed_history is not None:
                data_scaled = np.vstack([self.seed_history, data_scaled])
                offset = len(self.seed_history)
            else:
                offset = 0

            # Sliding window predictions
            X_batch = []
            valid_indices = []

            start_idx = offset if self.seed_history is not None else look_back - 1
            for i in range(start_idx, len(data_scaled)):
                X_batch.append(data_scaled[i - look_back + 1: i + 1])
                valid_indices.append(i - offset)  # map back to original df index

            if not X_batch:
                df['tma_predicted'] = np.nan
                df['status'] = 'Pending'
                return df

            X_batch = np.array(X_batch)

            # Predict
            preds = self.model.predict(X_batch, verbose=0)
            preds_scaled = preds[0] if isinstance(preds, list) else preds

            # Inverse transform
            preds_value = self.scaler_target.inverse_transform(preds_scaled).flatten()

            # Assign to df
            df['tma_predicted'] = np.nan
            df['status'] = 'Pending'

            for idx, pred_val in zip(valid_indices, preds_value):
                if 0 <= idx < len(df):
                    df.loc[df.index[idx], 'tma_predicted'] = float(pred_val)
                    df.loc[df.index[idx], 'status'] = (
                        "Bahaya" if pred_val >= threshold else "Normal"
                    )

            return df

        except Exception as e:
            print(f"Batch prediction error: {e}")
            import traceback
            traceback.print_exc()
            df['tma_predicted'] = np.nan
            df['status'] = 'Error'
            return df
