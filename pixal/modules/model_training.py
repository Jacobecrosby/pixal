import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras.losses import Huber

@register_keras_serializable()
def masked_mse_loss(y_true, y_pred):
    # Create a mask for non-zero values
    mask = tf.cast(tf.not_equal(y_true, 0.0), tf.float32)

    # Compute squared error only where y_true is non-zero
    squared_error = tf.square(y_true - y_pred) * mask

    # Normalize by the number of valid (non-zero) entries
    return tf.reduce_sum(squared_error) / tf.reduce_sum(mask)

def resolve_loss(name):
    lookup = {
    'masked_mse_loss': masked_mse_loss,
    'mse': 'mse',
    'mae': 'mae',
    'huber': Huber(delta=1.0)
        }
    return lookup.get(name, 'mse')


def _reduce_mse(err):  # err: (..., )
    return tf.reduce_mean(tf.square(err))

def _reduce_mae(err):
    return tf.reduce_mean(tf.abs(err))

def _reduce_rmse(err):
    return tf.sqrt(tf.reduce_mean(tf.square(err)) + 1e-12)

def make_channel_metric(idx: int, name: str, reducer=_reduce_mse):
    """
    Returns a tf.keras metric function computing a per-channel reduction.
    idx: channel index in the last dimension
    name: metric name shown in logs
    reducer: one of _reduce_mse / _reduce_mae / _reduce_rmse or your own
    """
    def metric(y_true, y_pred):
        ch_true = y_true[..., idx]
        ch_pred = y_pred[..., idx]
        return reducer(ch_true - ch_pred)
    metric.__name__ = name  # important so Keras logs the right name
    return metric

# ---------- common reshape ----------
def _reshape_to_PC(t, C: int):
    """Reshape last dim to (..., P, C); works for last dim == C (P=1) or flattened P*C."""
    D = tf.shape(t)[-1]
    c = tf.constant(C, tf.int32)
    with tf.control_dependencies([
        tf.debugging.assert_equal(D % c, 0, message="Last dim must be divisible by num channels")
    ]):
        P = D // c
    new_shape = tf.concat([tf.shape(t)[:-1], tf.stack([P, c])], axis=0)
    return tf.reshape(t, new_shape), P

# ---------- weighted losses (works for flattened or channel-last) ----------
def make_weighted_loss(channel_names, weights, base="mse", delta=1.0, mask_fn=None):
    C = len(channel_names)
    if len(weights) != C:
        raise ValueError("weights list length must match channel_names length")
    w_vec = tf.constant(weights, tf.float32)  # (C,)

    def _elem_loss(diff):
        if base == "mse":
            return tf.square(diff)
        if base == "mae":
            return tf.abs(diff)
        if base == "huber":
            abs_err = tf.abs(diff)
            quadratic = tf.minimum(abs_err, delta)
            linear = abs_err - quadratic
            return 0.5 * tf.square(quadratic) + delta * linear
        if base == "charbonnier":
            eps = 1e-3
            return tf.sqrt(tf.square(diff) + eps*eps) - eps
        raise ValueError("base must be in {'mse','mae','huber','charbonnier'}")

    def loss(y_true, y_pred):
        diff = y_true - y_pred                          # (..., D)
        diff_pc, P = _reshape_to_PC(diff, C)            # (..., P, C)
        per_elem = _elem_loss(diff_pc)                  # (..., P, C)
        per_elem = per_elem * w_vec                     # broadcast on C
        if mask_fn is not None:
            mask = tf.cast(mask_fn(y_true, y_pred), tf.float32)
            mask_pc, _ = _reshape_to_PC(mask, C)        # (..., P, C)
            per_elem = per_elem * mask_pc
            denom = tf.reduce_sum(mask_pc) + 1e-12
            return tf.reduce_sum(per_elem) / denom
        return tf.reduce_mean(per_elem)

    loss.__name__ = f"weighted_{base}"
    return loss

# ---------- per-channel metrics (handles flattened or channel-last) ----------
def make_per_channel_metric(idx: int, channel_names, reducer="mse"):
    """
    Logs per-channel error for arbitrary shapes. reducer in {'mse','mae','rmse'}.
    """
    C = len(channel_names)
    name = f"{reducer}_{channel_names[idx]}"

    def metric(y_true, y_pred):
        diff = y_true - y_pred
        diff_pc, _ = _reshape_to_PC(diff, C)            # (..., P, C)
        ch = diff_pc[..., idx]                          # (..., P)
        if reducer == "mse":
            val = tf.reduce_mean(tf.square(ch))
        elif reducer == "mae":
            val = tf.reduce_mean(tf.abs(ch))
        elif reducer == "rmse":
            val = tf.sqrt(tf.reduce_mean(tf.square(ch)) + 1e-12)
        else:
            raise ValueError("reducer must be in {'mse','mae','rmse'}")
        return val

    metric.__name__ = name
    return metric

# ---------- (optional) total weighted metric mirroring the loss ----------
def make_total_weighted_metric(channel_names, weights, base="mse", delta=1.0):
    C = len(channel_names)
    w_vec = tf.constant(weights, tf.float32)

    def metric(y_true, y_pred):
        diff = y_true - y_pred
        diff_pc, _ = _reshape_to_PC(diff, C)
        if base == "mse":
            per_elem = tf.square(diff_pc)
        elif base == "mae":
            per_elem = tf.abs(diff_pc)
        elif base == "huber":
            abs_err = tf.abs(diff_pc)
            quadratic = tf.minimum(abs_err, delta)
            linear = abs_err - quadratic
            per_elem = 0.5 * tf.square(quadratic) + delta * linear
        elif base == "charbonnier":
            eps = 1e-3
            per_elem = tf.sqrt(tf.square(diff_pc) + eps*eps) - eps
        else:
            raise ValueError
        per_elem = per_elem * w_vec
        return tf.reduce_mean(per_elem)

    metric.__name__ = f"weighted_{base}_metric"
    return metric

def make_weighted_contrib_metric(idx, channel_names, weights, base="huber", delta=0.5):
    C = len(channel_names)
    w = tf.constant(weights, tf.float32)[idx]

    def metric(y_true, y_pred):
        # reshape (..., P, C)
        D = tf.shape(y_true)[-1]
        P = D // C
        y_t = tf.reshape(y_true, tf.concat([tf.shape(y_true)[:-1], [P, C]], 0))
        y_p = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:-1], [P, C]], 0))
        d = y_t - y_p
        if base == "huber":
            ae = tf.abs(d[..., idx])
            q = tf.minimum(ae, delta)
            l = ae - q
            per_elem = 0.5*tf.square(q) + delta*l
        else:
            per_elem = tf.square(d[..., idx])  # MSE fallback
        return tf.reduce_mean(w * per_elem)

    metric.__name__ = f"wcontrib_{channel_names[idx]}"
    return metric


def make_channel_last_metric(idx, name, reducer=_reduce_mse):
    def metric(y_true, y_pred):
        return reducer(y_true[..., idx] - y_pred[..., idx])
    metric.__name__ = name
    return metric

def make_flat_channel_metric(idx, num_channels, name, reducer=_reduce_mse):
    def metric(y_true, y_pred):
        diff = y_true - y_pred                       # (..., D=P*C)
        D = tf.shape(diff)[-1]
        C = tf.constant(num_channels, tf.int32)
        with tf.control_dependencies([tf.debugging.assert_equal(D % C, 0)]):
            P = D // C
        new_shape = tf.concat([tf.shape(diff)[:-1], tf.stack([P, C])], axis=0)
        diff_pc = tf.reshape(diff, new_shape)        # (..., P, C)
        ch = diff_pc[..., idx]                       # (..., P)
        return reducer(ch)
    metric.__name__ = name
    return metric

def build_channel_metrics(channel_names, flattened=False, reducers=("mse",)):
    name_to_reducer = {"mse": _reduce_mse, "mae": _reduce_mae, "rmse": _reduce_rmse}
    C = len(channel_names)
    mets = []
    for i, ch in enumerate(channel_names):
        for r in reducers:
            reducer = name_to_reducer[r]
            name = f"{r}_{ch}"
            if flattened:
                mets.append(make_flat_channel_metric(i, C, name, reducer))
            else:
                mets.append(make_channel_last_metric(i, name, reducer))
    return mets
