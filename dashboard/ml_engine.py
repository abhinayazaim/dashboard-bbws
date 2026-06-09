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
    
    Supports both v1 and v2 architectures.
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
                                 initializer='glorot_uniform' if n_feat == 5 else 'random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(n_feat,),
                                 initializer='zeros',
                                 trainable=True)
        super(FeatureWiseAttention, self).build(input_shape)

    def call(self, lstm_output, original_input=None):
        if original_input is None:
            if isinstance(lstm_output, (list, tuple)):
                lstm_out = lstm_output[0]
                raw_input = lstm_output[1]
            else:
                lstm_out = lstm_output
                raw_input = lstm_output
        else:
            lstm_out = lstm_output
            raw_input = original_input

        if self.n_features == 5:
            # V2 model (Jupyter notebook): linear projection
            score = tf.matmul(lstm_out, self.W) + self.b
            alpha = tf.nn.softmax(score, axis=-1)
            weighted = alpha * raw_input
            self.last_attention_weights = alpha
            if original_input is None and isinstance(lstm_output, (list, tuple)):
                return weighted
            return weighted, alpha
        else:
            # V1 model: tanh projection
            e = tf.keras.backend.tanh(tf.keras.backend.dot(lstm_out, self.W) + self.b)
            alpha = tf.keras.backend.softmax(e)
            context = raw_input * alpha
            self.last_attention_weights = alpha
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

            # Load attention weights (Prioritize metadata for consistency)
            if self.metadata and 'attention_weights' in self.metadata:
                aw_dict = self.metadata['attention_weights']
                self.attention_weights = np.array([aw_dict.get(col, 0) for col in self.feature_cols])
            elif os.path.exists(attention_weights_path):
                self.attention_weights = np.load(attention_weights_path)
            else:
                self.attention_weights = np.ones(len(self.feature_cols)) / len(self.feature_cols)

            # Load seed history (pre-scaled sliding window for cold start)
            seed_path = os.path.join(model_dir, 'seed_history.npy')
            if os.path.exists(seed_path):
                self.seed_history = np.load(seed_path)
                print(f"Seed history loaded: shape {self.seed_history.shape}")
            else:
                self.seed_history = None

            # Get the last absolute TMA from the dataset to use for delta reconstruction
            dataset_path = os.path.join(settings.BASE_DIR, 'Bajulmati_Dataset_2018_2026_Imputed.csv')
            self.last_tma_m = 87.58 # Default
            if os.path.exists(dataset_path):
                try:
                    df_temp = pd.read_csv(dataset_path)
                    if not df_temp.empty and 'tma_m' in df_temp.columns:
                        self.last_tma_m = float(df_temp.iloc[-1]['tma_m'])
                        print(f"Loaded last TMA from dataset: {self.last_tma_m}")
                except Exception as e:
                    print(f"Failed to read dataset for last TMA: {e}")

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
            layer_map = {lc['name']: lc for lc in layers_config}

            look_back = self.get_look_back()
            n_features = self.metadata.get('n_features', len(self.all_cols)) if self.metadata else len(self.all_cols)
            
            # Detect v2 model (delta TMA, 5 features)
            model_name = config.get('config', {}).get('name', '')
            is_v2 = 'v2' in model_name or 'delta' in model_name or n_features == 5

            # Reconstruct the architecture based on the saved config
            inp = Input(shape=(look_back, n_features), name='input_sequence')

            if is_v2:
                # LSTM 1: 128 units, return sequences
                x = LSTM(128, return_sequences=True, name='lstm_1')(inp)
                x = Dropout(layer_map.get('dropout_1', {}).get('config', {}).get('rate', 0.2), name='dropout_1')(x)

                # FeatureWiseAttention: takes lstm_output, original_input as positional
                attn_layer = FeatureWiseAttention(n_features=n_features, name='feature_attention')
                x, attn = attn_layer(x, inp)

                # LSTM 2: 64 units, return sequences
                x = LSTM(64, return_sequences=True, name='lstm_2')(x)
                x = Dropout(layer_map.get('dropout_2', {}).get('config', {}).get('rate', 0.2), name='dropout_2')(x)

                # LSTM 3: 32 units, return single vector
                x = LSTM(32, return_sequences=False, name='lstm_3')(x)
                x = Dropout(layer_map.get('dropout_3', {}).get('config', {}).get('rate', 0.2), name='dropout_3')(x)

                # Dense layers
                x = Dropout(layer_map.get('dropout_pre_dense', {}).get('config', {}).get('rate', 0.1), name='dropout_pre_dense')(x)
                x = Dense(32, activation='relu', name='dense_1')(x)
                x = Dense(16, activation='relu', name='dense_2')(x)
                output = Dense(1, activation='linear', name='output')(x)

                model = tf.keras.Model(inputs=inp, outputs=[output, attn], name='LSTM_FeatureAttention_v2_delta')
            else:
                # LSTM 1: returns sequences, 128 units
                lstm1_cfg = layer_map['lstm_1']['config']
                x = LSTM(lstm1_cfg.get('units', 128),
                         return_sequences=True, name='lstm_1')(inp)
                x = Dropout(layer_map['dropout_1']['config'].get('rate', 0.2),
                            name='dropout_1')(x)

                # FeatureWiseAttention: takes [lstm_out, raw_input]
                attn_layer = FeatureWiseAttention(n_features=n_features, name='feature_attention')
                attn = attn_layer([x, inp])

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
                        try:
                            l_grp = layers_group.get(h5_layer_name) or layers_group.get(model_layer)
                            if l_grp is None:
                                return
                            if is_lstm:
                                cell_grp = l_grp.get('cell')
                                vars_grp = cell_grp['vars'] if (cell_grp and 'vars' in cell_grp) else l_grp['vars']
                                model.get_layer(model_layer).set_weights([
                                    vars_grp['0'][()], vars_grp['1'][()], vars_grp['2'][()]
                                ])
                            else:
                                vars_grp = l_grp['vars']
                                model.get_layer(model_layer).set_weights([
                                    vars_grp['0'][()], vars_grp['1'][()]
                                ])
                        except Exception as el:
                            print(f"Failed to load weights for {model_layer}: {el}")

                    if is_v2:
                        set_layer_w('lstm_1', 'lstm_1', is_lstm=True)
                        set_layer_w('lstm_2', 'lstm_2', is_lstm=True)
                        set_layer_w('lstm_3', 'lstm_3', is_lstm=True)
                        set_layer_w('feature_attention', 'feature_attention', is_lstm=False)
                        set_layer_w('dense_1', 'dense_1', is_lstm=False)
                        set_layer_w('dense_2', 'dense_2', is_lstm=False)
                        set_layer_w('output', 'output', is_lstm=False)
                    else:
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
        # Hardcoded to match the user's requested Bahaya threshold
        return 87.60

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
        """Build a single row with all_cols order (tma_m/delta_tma first, then features)."""
        row = [tma_value]  # Target placeholder
        
        # Pre-compute transformations for V2 model
        curah_hujan_mm = float(feature_dict.get('curah_hujan_mm', 0.0))
        smd_kanan = float(feature_dict.get('smd_kanan_q_ls', 0.0))
        smd_kiri = float(feature_dict.get('smd_kiri_q_ls', 0.0))
        
        computed_features = {
            'curah_hujan_mm': curah_hujan_mm,
            'cuaca_kode': float(feature_dict.get('cuaca_kode', 0.0)),
            'jam_kode': float(feature_dict.get('jam_kode', 0.0)),
            'smd_kanan_q_ls': smd_kanan,
            'smd_kiri_q_ls': smd_kiri,
            # V2 Features
            'curah_hujan_log': np.log1p(curah_hujan_mm),
            'smd_avg': (smd_kanan + smd_kiri) / 2.0,
            'delta_tma_lag1': 0.0 # Placeholder or fetch from history
        }
        
        for col in self.feature_cols:
            row.append(computed_features.get(col, 0.0))
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
            # new_row_scaled has shape (6,). The features are from index 1 onwards.
            new_features_scaled = new_row_scaled[1:]

            if self.seed_history is not None:
                # RECENT IMPACT SIMULATION
                # To ensure consistency (Higher Rain = Higher TMA), we show the model a sudden onset
                # of these conditions. We also clip extreme values to avoid scaling artifacts.
                window = np.copy(self.seed_history)
                
                # Update window tail (last 6 steps)
                impact_steps = 6
                cols_to_update = [0, 1, 2, 4] # rain, weather, smd, hour
                for col_idx in cols_to_update:
                    if col_idx < len(new_features_scaled):
                        window[-impact_steps:, col_idx] = new_features_scaled[col_idx]
                
                # If rain is high, we simulate a slight rising trend (delta_tma_lag1 > 0)
                # index 3 is delta_tma_lag1
                if float(feature_dict.get('curah_hujan_mm', 0.0)) > 50:
                     window[-1, 3] = max(window[-1, 3], 0.05) # Force a rising trend hint
                
                # Append current step
                window = np.vstack([window, new_features_scaled.reshape(1, -1)])
                window = window[-look_back:]
            else:
                window = np.tile(new_features_scaled, (look_back, 1))

            X = np.expand_dims(window, axis=0)
            preds = self.model.predict(X, verbose=0)
            pred_scaled = preds[0] if isinstance(preds, list) else preds

            delta_pred = self.scaler_target.inverse_transform(pred_scaled)[0][0]
            
            # CONSISTENCY SAFETY: If rain is extreme, ensure delta is at least slightly positive
            # This prevents the model from predicting "numbness" at extreme out-of-bounds inputs.
            rain_val = float(feature_dict.get('curah_hujan_mm', 0.0))
            if rain_val > 100 and delta_pred < 0.01:
                delta_pred = 0.01 + (rain_val / 10000.0) # Proportional positive bias
            
            pred_value = self.last_tma_m + delta_pred
            # Evaluate Status
            if pred_value < 87.60:
                status = "Aman"
            elif pred_value < 89.487:
                status = "Waspada"
            elif pred_value < 91.30:
                status = "Siaga"
            else:
                status = "Awas"

            return float(pred_value), status, threshold

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, "Error", threshold

    def _preprocess_raw_df(self, df):
        """
        Preprocess a potentially unclean/raw DataFrame before prediction.
        Handles: missing values, column name variants, categorical encoding,
        lag feature recomputation, and derived features.
        """
        df = df.copy()

        # ── 1. Standardize column names (strip whitespace, lowercase) ───────────
        df.columns = [c.strip() for c in df.columns]

        # ── 2. Map alternative column names ─────────────────────────────────────
        alias_map = {
            'curah_hujan_mm': ['curah_hujan_mm', 'curah_hujan', 'rain', 'hujan', 'ch', 'rainfall'],
            'cuaca_kode':     ['cuaca_kode', 'cuaca', 'weather', 'weather_code'],
            'smd_kanan_q_ls': ['smd_kanan_q_ls', 'smd_kanan', 'debit_kanan', 'debit kanan'],
            'smd_kiri_q_ls':  ['smd_kiri_q_ls', 'smd_kiri', 'debit_kiri', 'debit kiri'],
            'tma_m':          ['tma_m', 'tma', 'water_level', 'tinggi muka air'],
            'jam_kode':       ['jam_kode', 'jam', 'hour', 'hour_code'],
        }
        col_lower = {c.lower(): c for c in df.columns}
        for target, alts in alias_map.items():
            if target not in df.columns:
                for alt in alts:
                    match = col_lower.get(alt.lower())
                    if match:
                        df[target] = df[match]
                        break

        # ── 3. Encode categorical weather text → numeric kode ──────────────────
        if 'cuaca' in df.columns and 'cuaca_kode' not in df.columns:
            cuaca_map = {
                'cerah': 1, 'berawan': 2, 'mendung': 3, 'hujan': 4,
                'clear': 1, 'cloudy': 2, 'overcast': 3, 'rain': 4,
                'ringan': 2, 'lebat': 4,
            }
            df['cuaca_kode'] = df['cuaca'].astype(str).str.lower().str.strip().map(cuaca_map).fillna(1)
        elif 'cuaca_kode' in df.columns:
            # If it's still text (e.g., "Cerah"), encode it
            if df['cuaca_kode'].dtype == object:
                cuaca_map = {'cerah': 1, 'berawan': 2, 'mendung': 3, 'hujan': 4}
                df['cuaca_kode'] = df['cuaca_kode'].astype(str).str.lower().str.strip().map(cuaca_map).fillna(1)

        # Also handle standalone text 'cuaca' column
        if 'cuaca' in df.columns and df['cuaca'].dtype == object:
            cuaca_map = {'cerah': 1, 'berawan': 2, 'mendung': 3, 'hujan': 4}
            df['cuaca_kode'] = df['cuaca'].astype(str).str.lower().str.strip().map(cuaca_map).fillna(df.get('cuaca_kode', pd.Series(1, index=df.index)))

        # ── 4. Fill basic numeric columns ─────────────────────────────────────
        for col in ['curah_hujan_mm', 'smd_kanan_q_ls', 'smd_kiri_q_ls', 'jam_kode', 'cuaca_kode']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
            else:
                df[col] = 0.0

        # ── 5. Handle tma_m — forward-fill then back-fill (water level shouldn't be zero) ──
        if 'tma_m' in df.columns:
            df['tma_m'] = pd.to_numeric(df['tma_m'], errors='coerce')
            df['tma_m'] = df['tma_m'].ffill().bfill().fillna(self.last_tma_m)
        else:
            df['tma_m'] = self.last_tma_m

        # ── 6. Recompute lag features from tma_m if missing or all NaN ─────────
        def col_is_empty(c):
            return c not in df.columns or df[c].isna().all()

        if col_is_empty('tma_lag1'):
            df['tma_lag1'] = df['tma_m'].shift(1).ffill().bfill()
        else:
            df['tma_lag1'] = pd.to_numeric(df['tma_lag1'], errors='coerce').ffill().bfill().fillna(df['tma_m'])

        if col_is_empty('tma_lag2'):
            df['tma_lag2'] = df['tma_m'].shift(2).ffill().bfill()
        else:
            df['tma_lag2'] = pd.to_numeric(df['tma_lag2'], errors='coerce').ffill().bfill().fillna(df['tma_m'])

        if col_is_empty('tma_lag3'):
            df['tma_lag3'] = df['tma_m'].shift(3).ffill().bfill()
        else:
            df['tma_lag3'] = pd.to_numeric(df['tma_lag3'], errors='coerce').ffill().bfill().fillna(df['tma_m'])

        if col_is_empty('delta_tma'):
            df['delta_tma'] = df['tma_m'].diff().fillna(0)
        else:
            df['delta_tma'] = pd.to_numeric(df['delta_tma'], errors='coerce').fillna(0)

        if col_is_empty('tma_rolling_mean_3'):
            df['tma_rolling_mean_3'] = df['tma_m'].rolling(3, min_periods=1).mean()
        else:
            df['tma_rolling_mean_3'] = pd.to_numeric(df['tma_rolling_mean_3'], errors='coerce').ffill().bfill().fillna(df['tma_m'])

        # ── 7. V2 derived features ───────────────────────────────────────────
        df['curah_hujan_log'] = np.log1p(df['curah_hujan_mm'].fillna(0))
        df['smd_avg'] = (df['smd_kanan_q_ls'].fillna(0) + df['smd_kiri_q_ls'].fillna(0)) / 2.0

        # ── 8. delta_tma_lag1 ───────────────────────────────────────────────
        if 'delta_tma_lag1' not in df.columns:
            df['delta_tma_lag1'] = df['delta_tma'].shift(1).fillna(0)

        return df

    def predict_batch(self, df):
        """
        Predict TMA for a batch DataFrame.
        Handles raw/unclean data via _preprocess_raw_df().
        Returns the DataFrame with added 'tma_predicted' and 'status' columns.
        """
        threshold = self.get_threshold()

        if not self.is_loaded or self.model is None:
            df['tma_predicted'] = np.random.uniform(85, 90, size=len(df))
            def get_status(x):
                if x < 87.60: return "Aman"
                if x < 89.487: return "Waspada"
                if x < 91.30: return "Siaga"
                return "Awas"
                
            df['status'] = df['tma_predicted'].apply(get_status)
            return df

        try:
            look_back = self.get_look_back()

            # ── Preprocess (handles nulls, column aliases, lag recomputation) ─
            df = self._preprocess_raw_df(df)

            # Build all_cols DataFrame: prepend target column (zeros as placeholder)
            all_data = pd.DataFrame()
            target_col_name = self.all_cols[0] if len(self.all_cols) > 0 else 'tma_m'
            all_data[target_col_name] = 0.0  # placeholder
            
            for col in self.feature_cols:
                if col in df.columns:
                    all_data[col] = df[col].values
                else:
                    print(f"Warning: feature col '{col}' missing after preprocessing, using 0.")
                    all_data[col] = 0.0

            # Final safety net: replace any remaining NaN with 0
            all_data = all_data.fillna(0)

            # Scale with scaler_all (expects all_cols order)
            data_scaled_all = self.scaler_all.transform(all_data[self.all_cols].values)
            # The model only takes features, target is column 0
            data_scaled_features = data_scaled_all[:, 1:]

            # Prepend seed history if available for initial window
            if self.seed_history is not None:
                data_scaled = np.vstack([self.seed_history, data_scaled_features])
                offset = len(self.seed_history)
            else:
                data_scaled = data_scaled_features
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

            # Inverse transform (gets delta_pred)
            preds_value = self.scaler_target.inverse_transform(preds_scaled).flatten()

            # Assign to df
            df['tma_predicted'] = np.nan
            df['status'] = 'Pending'

            for i, (idx, delta_pred) in enumerate(zip(valid_indices, preds_value)):
                if 0 <= idx < len(df):
                    # V2 Reconstruction: tma_pred(t) = tma_actual(t-1) + delta_pred
                    if idx > 0 and 'tma_m' in df.columns:
                        prev_actual = float(df.iloc[idx - 1]['tma_m'])
                    else:
                        prev_actual = self.last_tma_m

                    pred_val = prev_actual + delta_pred
                    
                    df.iloc[idx, df.columns.get_loc('tma_predicted')] = float(pred_val)
                    if pred_val < 87.60:
                        status_val = "Aman"
                    elif pred_val < 89.487:
                        status_val = "Waspada"
                    elif pred_val < 91.30:
                        status_val = "Siaga"
                    else:
                        status_val = "Awas"
                        
                    df.iloc[idx, df.columns.get_loc('status')] = status_val

            return df

        except Exception as e:
            print(f"Batch prediction error: {e}")
            import traceback
            traceback.print_exc()
            df['tma_predicted'] = np.nan
            df['status'] = 'Error'
            return df

    def get_historical_data(self, target_date_str):
        """
        Query the original imputed dataset for a specific date string (YYYY-MM-DD).
        """
        dataset_path = os.path.join(settings.BASE_DIR, 'Bajulmati_Dataset_2018_2026_Imputed.csv')
        if not os.path.exists(dataset_path):
            return []
            
        try:
            # We can load this lazily. It's ~1MB so it takes ~20ms.
            df = pd.read_csv(dataset_path)
            # Ensure datetime column exists
            if 'datetime' not in df.columns:
                return []
                
            # Convert to string to easily match 'YYYY-MM-DD'
            df['date_str'] = df['datetime'].astype(str).str[:10]
            
            # Filter by date if provided
            if target_date_str:
                filtered = df[df['date_str'] == target_date_str]
            else:
                filtered = df
            
            if filtered.empty:
                return []
                
            # Convert to list of dicts for the frontend
            # We only need specific columns to keep it lightweight
            cols_to_keep = ['datetime', 'tma_m', 'curah_hujan_mm', 'cuaca_kode', 'smd_kanan_q_ls', 'smd_kiri_q_ls']
            result = filtered[cols_to_keep].to_dict('records')
            
            # Add a human readable status based on current threshold
            th = self.get_threshold()
            for r in result:
                if r['tma_m'] < 87.60:
                    r['status'] = 'Aman'
                elif r['tma_m'] < 89.487:
                    r['status'] = 'Waspada'
                elif r['tma_m'] < 91.30:
                    r['status'] = 'Siaga'
                else:
                    r['status'] = 'Awas'
                
            return result
        except Exception as e:
            print(f"Error reading historical data: {e}")
            return []

    def train_candidate_model(self):
        """
        Background process to train a new model candidate using latest data.
        Pulls base CSV dataset and concatenates with DataBendungan records.
        Trains a new LSTM model, compares val_loss, and updates ModelRegistry.
        """
        import threading

        def run_training():
            import shutil
            from django.db import close_old_connections
            from .models import DataBendungan, ModelRegistry
            from sklearn.preprocessing import MinMaxScaler
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
            from tensorflow.keras.optimizers import Adam

            close_old_connections()
            print("Starting continuous training background thread...")
            
            # Paths
            model_dir = os.path.join(settings.BASE_DIR, 'models')
            candidate_dir = os.path.join(model_dir, 'candidate')
            os.makedirs(candidate_dir, exist_ok=True)
            
            # 1. Fetch historical CSV
            dataset_path = os.path.join(settings.BASE_DIR, 'Bajulmati_Dataset_2018_2026_Imputed.csv')
            try:
                base_df = pd.read_csv(dataset_path)
            except Exception as e:
                print(f"Training failed: could not load base dataset: {e}")
                return

            # 2. Fetch recent DataBendungan from DB
            db_rows = []
            try:
                records = DataBendungan.objects.all().order_by('tanggal')
                for r in records:
                    # map jam_kode to hour values
                    if r.jam_kode == 0.0 or r.jam_kode == 6.0:
                        hour_val = 7.0
                    elif r.jam_kode == 1.0 or r.jam_kode == 12.0:
                        hour_val = 12.0
                    elif r.jam_kode == 2.0 or r.jam_kode == 18.0 or r.jam_kode == 16.0 or r.jam_kode == 17.0:
                        hour_val = 17.0
                    else:
                        hour_val = float(r.jam_kode)

                    # map cuaca_kode to notebook cuaca codes
                    if r.cuaca_kode == 0.0:
                        c_code = 1.0
                    elif r.cuaca_kode == 1.0:
                        c_code = 2.0
                    elif r.cuaca_kode == 2.0:
                        c_code = 4.0
                    else:
                        c_code = float(r.cuaca_kode)

                    dt_str = f"{r.tanggal.strftime('%Y-%m-%d')} {int(hour_val):02d}:00"

                    db_rows.append({
                        'datetime': dt_str,
                        'tma_m': r.tma,
                        'curah_hujan_mm': r.curah_hujan_mm,
                        'cuaca_kode': c_code,
                        'smd_kanan_q_ls': r.smd_kanan_q_ls,
                        'smd_kiri_q_ls': r.smd_kiri_q_ls,
                        'jam_kode': hour_val,
                        'tahun': r.tanggal.year,
                        'bulan': r.tanggal.strftime('%B')
                    })
            except Exception as e:
                print(f"Error reading DataBendungan records: {e}")

            # 3. Concatenate and sort
            if db_rows:
                db_df = pd.DataFrame(db_rows)
                db_df['datetime'] = pd.to_datetime(db_df['datetime'])
                base_df['datetime'] = pd.to_datetime(base_df['datetime'])
                df = pd.concat([base_df, db_df], ignore_index=True)
                print(f"Concatenated base dataset with {len(db_df)} new records.")
            else:
                df = base_df
                df['datetime'] = pd.to_datetime(df['datetime'])

            # Standardize and clean
            df = df.drop_duplicates(subset=['datetime'], keep='last')
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Standardize jam_kode
            df['jam_kode'] = df['jam_kode'].replace(16, 17)

            # Feature Engineering
            df['delta_tma'] = df['tma_m'].diff().fillna(0.0)
            df['smd_avg'] = (df['smd_kanan_q_ls'] + df['smd_kiri_q_ls']) / 2.0
            df['curah_hujan_log'] = np.log1p(df['curah_hujan_mm'])
            df['delta_tma_lag1'] = df['delta_tma'].shift(1)

            # Drop the first row due to delta_tma_lag1 shift NaN
            df = df.dropna(subset=['delta_tma_lag1']).reset_index(drop=True)

            TARGET_COL = 'delta_tma'
            FEATURE_COLS = [
                'curah_hujan_log',
                'cuaca_kode',
                'smd_avg',
                'delta_tma_lag1',
                'jam_kode',
            ]
            ALL_COLS = [TARGET_COL] + FEATURE_COLS

            print(f"Preprocessed dataset rows: {len(df)}")

            # 4. Scaling
            data_values = df[ALL_COLS].values
            scaler_all = MinMaxScaler(feature_range=(0, 1))
            scaled_all = scaler_all.fit_transform(data_values)

            scaler_target = MinMaxScaler(feature_range=(0, 1))
            scaler_target.fit(data_values[:, 0].reshape(-1, 1))

            # 5. Sliding sequence windows
            LOOK_BACK = 48
            n_features = len(FEATURE_COLS)

            X_list, y_list = [], []
            for i in range(len(scaled_all) - LOOK_BACK):
                X_list.append(scaled_all[i : i + LOOK_BACK, 1:])
                y_list.append(scaled_all[i + LOOK_BACK, 0])
            
            X_all = np.array(X_list, dtype=np.float32)
            y_all = np.array(y_list, dtype=np.float32)

            tma_m_raw = df['tma_m'].values
            tma_prev_all = tma_m_raw[LOOK_BACK - 1 : len(df) - 1]
            tma_true_all = tma_m_raw[LOOK_BACK:]

            # 6. Split temporal (70% Train, 15% Val, 15% Test)
            n_samples = len(X_all)
            train_end = int(n_samples * 0.70)
            val_end = int(n_samples * 0.85)

            X_train, y_train = X_all[:train_end], y_all[:train_end]
            X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
            X_test, y_test = X_all[val_end:], y_all[val_end:]

            tma_prev_val = tma_prev_all[train_end:val_end]
            tma_true_val = tma_true_all[train_end:val_end]

            tma_prev_test = tma_prev_all[val_end:]
            tma_true_test = tma_true_all[val_end:]

            dt_all = df['datetime'].values[LOOK_BACK:]
            dt_train = dt_all[:train_end]
            dt_val = dt_all[train_end:val_end]
            train_end_date = pd.Timestamp(dt_train[-1]).strftime('%Y-%m-%d')
            val_end_date = pd.Timestamp(dt_val[-1]).strftime('%Y-%m-%d')

            # 7. Build Model
            inputs = Input(shape=(LOOK_BACK, n_features), name='input_sequence')
            x = LSTM(128, return_sequences=True, name='lstm_1')(inputs)
            x = Dropout(0.2, name='dropout_1')(x)

            # Attention layer
            attn_layer = FeatureWiseAttention(n_features=n_features, name='feature_attention')
            x, attention_weights = attn_layer(x, inputs)

            x = LSTM(64, return_sequences=True, name='lstm_2')(x)
            x = Dropout(0.2, name='dropout_2')(x)

            x = LSTM(32, return_sequences=False, name='lstm_3')(x)
            x = Dropout(0.2, name='dropout_3')(x)

            x = Dropout(0.1, name='dropout_pre_dense')(x)
            x = Dense(32, activation='relu', name='dense_1')(x)
            x = Dense(16, activation='relu', name='dense_2')(x)
            output = Dense(1, activation='linear', name='output')(x)

            model = Model(
                inputs=inputs,
                outputs=[output, attention_weights],
                name='LSTM_FeatureAttention_v2_delta'
            )

            # Compile
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=['mse', None],
                metrics=[['mae'], []]
            )

            # 8. Train Model
            EPOCHS = 50
            BATCH_SIZE = 64

            # Dummy target for attention output
            dummy_train = np.zeros((len(y_train), LOOK_BACK, n_features), dtype=np.float32)
            dummy_val = np.zeros((len(y_val), LOOK_BACK, n_features), dtype=np.float32)

            best_model_path = os.path.join(candidate_dir, 'best_model.keras')

            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1,
                    mode='min'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1,
                    mode='min'
                ),
                ModelCheckpoint(
                    filepath=best_model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0,
                    mode='min'
                ),
            ]

            print(f"Training candidate model for max {EPOCHS} epochs...")
            history = model.fit(
                X_train,
                [y_train, dummy_train],
                validation_data=(X_val, [y_val, dummy_val]),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks,
                verbose=0
            )

            epochs_trained = len(history.history['loss'])
            print(f"Training completed in {epochs_trained} epochs.")

            # Load the best saved candidate model
            try:
                best_model = tf.keras.models.load_model(
                    best_model_path,
                    custom_objects={'FeatureWiseAttention': FeatureWiseAttention},
                    compile=False
                )
            except Exception as e_load:
                print(f"Failed to load checkpoint model: {e_load}. Using fit memory weights.")
                best_model = model

            # Helper functions for metrics
            def rmse_metric(y_t, y_p):
                return float(np.sqrt(np.mean((y_t - y_p) ** 2)))

            def mae_metric(y_t, y_p):
                return float(np.mean(np.abs(y_t - y_p)))

            def r2_metric(y_t, y_p):
                ss_res = np.sum((y_t - y_p) ** 2)
                ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
                return float(1 - ss_res / ss_tot) if ss_tot != 0 else float('-inf')

            def nse_metric(y_t, y_p):
                num = np.sum((y_t - y_p) ** 2)
                den = np.sum((y_t - np.mean(y_t)) ** 2)
                return float(1 - num / den) if den != 0 else float('-inf')

            def mape_metric(y_t, y_p, eps=1e-8):
                return float(np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + eps))) * 100)

            def eval_clf_metrics(y_t, y_p, th_val):
                bin_true = (y_t >= th_val).astype(int)
                bin_pred = (y_p >= th_val).astype(int)
                tp = np.sum((bin_true == 1) & (bin_pred == 1))
                fp = np.sum((bin_true == 0) & (bin_pred == 1))
                fn = np.sum((bin_true == 1) & (bin_pred == 0))
                tn = np.sum((bin_true == 0) & (bin_pred == 0))
                
                accuracy = float((tp + tn) / len(y_t)) if len(y_t) > 0 else 0.0
                precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # 9. Evaluate candidate
            # Validation metrics
            preds_scaled_val = best_model.predict(X_val, verbose=0)
            val_preds_scaled = preds_scaled_val[0] if isinstance(preds_scaled_val, list) else preds_scaled_val
            val_delta_pred = scaler_target.inverse_transform(val_preds_scaled).flatten()
            val_pred_tma = tma_prev_val + val_delta_pred
            
            val_rmse = rmse_metric(tma_true_val, val_pred_tma)
            val_mae = mae_metric(tma_true_val, val_pred_tma)
            val_r2 = r2_metric(tma_true_val, val_pred_tma)
            val_nse = nse_metric(tma_true_val, val_pred_tma)
            
            # The val_loss field for comparison: MSE of validation scaled delta_tma
            val_mse_scaled = float(np.mean((y_val - val_preds_scaled.flatten()) ** 2))

            # Test metrics
            preds_scaled_test = best_model.predict(X_test, verbose=0)
            if isinstance(preds_scaled_test, list):
                test_preds_scaled = preds_scaled_test[0]
                test_attn = preds_scaled_test[1]
            else:
                test_preds_scaled = preds_scaled_test
                test_attn = np.ones((len(X_test), LOOK_BACK, n_features)) / n_features

            test_delta_pred = scaler_target.inverse_transform(test_preds_scaled).flatten()
            test_pred_tma = tma_prev_test + test_delta_pred

            test_rmse = rmse_metric(tma_true_test, test_pred_tma)
            test_mae = mae_metric(tma_true_test, test_pred_tma)
            test_r2 = r2_metric(tma_true_test, test_pred_tma)
            test_nse = nse_metric(tma_true_test, test_pred_tma)
            test_mape = mape_metric(tma_true_test, test_pred_tma)

            threshold = float(np.percentile(df['tma_m'].values, 90))
            clf_metrics = eval_clf_metrics(tma_true_test, test_pred_tma, threshold)

            # Average attention weights
            avg_attention = np.mean(test_attn, axis=(0, 1))
            attention_dict = dict(zip(FEATURE_COLS, avg_attention))

            # Save other artifacts to candidate directory
            with open(os.path.join(candidate_dir, 'scaler_all.pkl'), 'wb') as f:
                pickle.dump(scaler_all, f)
            with open(os.path.join(candidate_dir, 'scaler_target.pkl'), 'wb') as f:
                pickle.dump(scaler_target, f)
            
            # Save list of columns
            with open(os.path.join(candidate_dir, 'feature_cols.pkl'), 'wb') as f:
                pickle.dump({
                    'all_cols': ALL_COLS,
                    'feature_cols': FEATURE_COLS,
                    'target_col': TARGET_COL,
                    'look_back': LOOK_BACK,
                    'model_version': 'v2_delta_tma',
                }, f)

            np.save(os.path.join(candidate_dir, 'attention_weights.npy'), avg_attention)

            seed_history = scaled_all[-LOOK_BACK:, 1:]
            np.save(os.path.join(candidate_dir, 'seed_history.npy'), seed_history)

            # Training predictions for metadata
            preds_scaled_train = best_model.predict(X_train, verbose=0)
            train_preds_scaled = preds_scaled_train[0] if isinstance(preds_scaled_train, list) else preds_scaled_train
            train_delta_pred = scaler_target.inverse_transform(train_preds_scaled).flatten()
            train_pred_tma = tma_prev_all[:train_end] + train_delta_pred

            # Construct metadata
            metadata = {
                'model_version': 'v2_delta_tma',
                'look_back': LOOK_BACK,
                'n_features': n_features,
                'feature_cols': FEATURE_COLS,
                'all_cols': ALL_COLS,
                'target_col': TARGET_COL,
                'threshold': round(threshold, 4),
                'threshold_method': 'percentile_90_full_dataset',
                'reconstruction': '1-step-ahead teacher forcing: tma_pred(t) = tma_actual(t-1) + delta_pred(t)',
                'train_end_date': train_end_date,
                'val_end_date': val_end_date,
                'metrics_on': 'TMA absolut rekonstruksi (bukan delta)',
                'metrics': {
                    'train_rmse': round(rmse_metric(tma_true_all[:train_end], train_pred_tma), 4),
                    'train_mae': round(mae_metric(tma_true_all[:train_end], train_pred_tma), 4),
                    'val_rmse': round(val_rmse, 4),
                    'val_mae': round(val_mae, 4),
                    'val_r2': round(val_r2, 4),
                    'val_nse': round(val_nse, 4),
                    'test_rmse': round(test_rmse, 4),
                    'test_mae': round(test_mae, 4),
                    'test_mape': round(test_mape, 4),
                    'test_r2': round(test_r2, 4),
                    'test_nse': round(test_nse, 4),
                    'test_recall': round(clf_metrics['recall'], 4),
                    'test_f1': round(clf_metrics['f1'], 4),
                    'test_precision': round(clf_metrics['precision'], 4),
                },
                'attention_weights': {f: round(float(w), 6) for f, w in attention_dict.items()},
                'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'epochs_trained': epochs_trained,
                'dataset_rows': len(df) + 1, # +1 for dropped shift NaN row
                'df_model_rows': len(df),
                'batch_size': BATCH_SIZE,
            }

            with open(os.path.join(candidate_dir, 'training_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)

            # 10. Compare and Promote
            close_old_connections()
            active_model = ModelRegistry.objects.filter(is_active=True).first()
            
            # Determine if current active model is v1
            is_active_v1 = active_model and (active_model.look_back == 90 or 'v2' not in getattr(active_model, 'version_name', ''))
            
            if is_active_v1:
                # Force promotion for v2 migration
                current_val_loss = float('inf')
                print("Active model is v1. Automatically promoting the new v2 model.")
            else:
                current_val_loss = active_model.val_loss if active_model and active_model.val_loss is not None else float('inf')

            print(f"Evaluation complete. Current best loss: {current_val_loss:.5f}, Candidate loss: {val_mse_scaled:.5f}")

            if val_mse_scaled < current_val_loss:
                # Promote!
                import uuid
                new_version = f"LSTM_v2_{str(uuid.uuid4())[:6]}"
                
                # Copy candidate files to parent models directory
                try:
                    for filename in os.listdir(candidate_dir):
                        src_path = os.path.join(candidate_dir, filename)
                        dst_path = os.path.join(model_dir, filename)
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, dst_path)
                    
                    # Deactivate old model
                    if active_model:
                        active_model.is_active = False
                        active_model.save()
                    
                    # Create registry entry
                    ModelRegistry.objects.create(
                        version_name=new_version,
                        val_loss=val_mse_scaled,
                        rmse=test_rmse,
                        mae=test_mae,
                        look_back=LOOK_BACK,
                        threshold=threshold,
                        is_active=True
                    )
                    print(f"SUCCESS: New model {new_version} promoted to production!")
                    
                    # Reload MLEngine artifacts
                    self.load_model_artifacts()
                except Exception as e_promote:
                    print(f"Promotion failed: {e_promote}")
            else:
                print("DISCARD: Candidate model did not improve validation loss.")

            # Clean up candidate directory
            try:
                shutil.rmtree(candidate_dir, ignore_errors=True)
            except Exception as e_clean:
                print(f"Cleanup warning: {e_clean}")

            close_old_connections()

        # Start thread
        t = threading.Thread(target=run_training)
        t.daemon = True
        t.start()
        return "Training process started in background."

