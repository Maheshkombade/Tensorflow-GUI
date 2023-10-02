import numpy as np
from sklearn.linear_model import LinearRegression

def get_slope(x1, y1, x2, y2):
    return (y2-y1) / (x2-x1)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def model_fit_inference(train_loss, val_loss):
    """
    Arguments
    =========

    train_loss - Array of loss for training
    val_loss - Array of loss for validation

    Returns
    =======

    A string containing all the inferences based on training and validation loss
    """
    SLOPE_THRES = 0.01
    SLOPE_LOOKBACK = 3 #fina 3
    SLOPE_AVG_THRES = -0.3
    EPS = 1e-7
    COFF_VAR_THRES = 0.3
    WINDOW = 5  #final=3

    msg = ""

    train_loss_orig = train_loss
    val_loss_orig = val_loss

    train_loss = moving_average(train_loss, WINDOW)
    val_loss = moving_average(val_loss, WINDOW)
    n_epochs = np.arange(1, len(train_loss) + 1)

    train_diff = train_loss_orig[WINDOW-1:] - train_loss
    val_diff = val_loss_orig[WINDOW-1:] - val_loss

    model = LinearRegression()
    model.fit(n_epochs.reshape(-1,1), train_loss)
    slope = model.coef_[0]

    if abs(slope) < SLOPE_THRES:
        msg += "Underfitting\n"
    else:
        msg += "Not Underfitting\n"

    train_slopes = []
    val_slopes = []

    for idx in range(-SLOPE_LOOKBACK, -2, +1):
        train_slopes.append(get_slope(n_epochs[idx], train_loss[idx], n_epochs[idx+1], train_loss[idx+1]))
        val_slopes.append(get_slope(n_epochs[idx], val_loss[idx], n_epochs[idx+1], val_loss[idx+1]))

    train_slope_avg = np.mean(train_slopes)
    val_slope_avg = np.mean(val_slopes)

    if train_slope_avg < 0 and val_slope_avg >= 0:
        if val_slope_avg - train_slope_avg > EPS:
            msg += "Overfitting\n"
        else:
            msg += "Not Overfitting\n"
    else:
        msg += "Not Overfitting\n"

    train_slopes = []
    for idx in range(-SLOPE_LOOKBACK, -2, +1):
        train_slopes.append(get_slope(n_epochs[idx], train_loss[idx], n_epochs[idx+1], train_loss[idx+1]))
    train_slope_avg = np.mean(train_slopes)

    if train_slope_avg > 0:
        msg += "Diverging\n"
    else:
        msg += "Not Diverging\n"

    train_coff_var = np.std(train_diff) / np.mean(train_diff)
    val_coff_var = np.std(val_diff) / np.mean(val_diff)

    if train_coff_var > COFF_VAR_THRES:
        msg += "Training loss has high variation.\n"
    else:
        msg += "Acceptable variance of training loss.\n"

    if val_coff_var > COFF_VAR_THRES:
        msg += "Validation loss has high variation.\n"
    else:
        msg += "Acceptable variance of validation loss.\n"

    return msg
