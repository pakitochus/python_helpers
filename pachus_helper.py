import numpy as np
import pandas as pd

def structConfMat(confmat, index=0):
    """
    Creates a pandas dataframe from the confusion matrix. It distinguishes
    between binary and multi-class classification. 

    Parameters
    ----------
    confmat : numpy.ndarray
        Array with n rows, each of one being a flattened confusion matrix.
    index : INT, optional
        Integer for index of the dataframe. The default is 0.

    Returns
    -------
    performance : pd.DataFrame
        Dataframe with all classification performance metrics.

    """
    conf = np.sum(confmat,axis=0)
    intdim = int(np.sqrt(confmat.shape[1]))
    conf = conf.reshape((intdim,intdim))
    N = np.float(np.sum(conf))
    cr = sum(np.diag(conf))/N
    crstd= (1.*np.sum(confmat[:,[0,-1]],axis=1)/np.sum(confmat,axis=1)).std()
    aux = 1.*confmat[:,[0,-1]]/np.vstack((np.sum(confmat[:,[0,2]],axis=1),np.sum(confmat[:,[1,3]],axis=1))).T
    
    performance = pd.DataFrame({'CorrectRate': cr, 'ErrorRate': 1-cr, 'CRstd': crstd}, index=[index])
    if confmat.shape[1]==4:
        sens = conf[1,-1]/np.sum(conf[-1])
        sensstd = np.nanstd(aux[:,-1])
        spec = conf[0,0]/np.sum(conf[0])
        specstd = np.nanstd(aux[:,])
        precision = sens/(sens+1-spec)
        f1 = 2*precision*sens/(precision+sens)
        b_acc = (sens+spec)/2
        auxperf = pd.DataFrame({'Sensitivity': sens, 'SensSTD': sensstd, 'Specificity': spec, 'SpecSTD': specstd, 'Precision':precision, 'f1':f1, 'balAcc':b_acc}, index=[index])
        performance = pd.concat((performance, auxperf), axis=1)
    else:
        b_acc = 0
        for ix in range(conf.shape[1]):
            fila = conf[ix]
            auxacc = fila[ix]/fila.sum()
            auxperf = pd.DataFrame({f'Class_{ix}': auxacc}, index=[index])
            b_acc += auxacc
            performance = pd.concat((performance, auxperf),axis=1)
        performance['balAcc'] = b_acc/ix
    return performance


def regressionMetrics(predicted, originals, index=0):
    """
    Creates regression performance metrics. 
    
    Parameters
    ----------
    predicted : list of arrays (one array per cv iteration)
        Predicted outcome of the regression model.
    originals : list of arrays (one array per cv iteration)
        Original outcome (true) of the regression model.
    index : INT, optional
        Integer for index of the dataframe. The default is 0.

    Returns
    -------
    performance : pd.DataFrame
        Dataframe with all the performance metrics.

    """
    from sklearn.metrics import r2_score
    MSE = []
    MAE = []
    R2 = []
    RMSE = []
    for ix in range(len(predicted)):
        MSE.append(np.mean((originals[ix] - predicted[ix])**2))
        MAE.append(np.mean(abs(originals[ix] - predicted[ix])))
        RMSE.append(np.sqrt(MSE[ix]))
        R2.append(r2_score(originals[ix], predicted[ix]))
    performance = pd.DataFrame({'MAE': np.mean(MAE), 'MAE (STD)': np.std(MAE),
                                'R2': np.mean(R2), 'R2 (STD)': np.std(R2),
                                'MSE': np.mean(MSE), 'MSE (STD)': np.std(MSE),
                                'RMSE': np.mean(RMSE), 'RMSE (STD)': np.std(RMSE)}, 
                               index=[index])
    return performance
