import math 
import numpy as np
from statistics import mean
from sklearn.metrics import mean_squared_error,  mean_absolute_error, r2_score


def mae( preds, target ):
    mae = mean_absolute_error( preds, target )
    print('MAE: {}'.format(mae))
    return mae


def r2( preds, target ):
    r2 = r2_score( preds, target )
    print('R2 Sore: {}'.format(r2))
    return r2


def rmse( preds, target ):
    rmse = math.sqrt( mean_squared_error(preds, target) )
    print('Mean value: {}. RMSE: {}'.format(mean(target),rmse))
    return rmse


def smape(preds, target):
    n = len(preds)
    # masked_arr = ~((preds == 0) & (target == 0))
    # preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    denom /= 2
    smape_val = ( 100* np.sum(num / denom)) / n
    print("SMAPE: ", smape_val)
    return smape_val


def print_acc_metrics( test_pred_df, feature ):
    print(f"{feature} accuracy")
    print("="*50)
    RMSE = rmse(test_pred_df[f"{feature}_pred"],  test_pred_df[feature])
    MAE = mae(test_pred_df[f"{feature}_pred"],  test_pred_df[feature])
    R2 = r2(test_pred_df[f"{feature}_pred"],  test_pred_df[feature])
    SMAPE = smape(test_pred_df[f"{feature}_pred"], test_pred_df[feature])

