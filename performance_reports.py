import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def structConfMat(confmat, index=0, multiple=False, useratio=False):
    """
    Creates a pandas dataframe from the confusion matrix. It distinguishes
    between binary and multi-class classification. 
    
    TODO: especificar formato exacto: corregir: ytrue - eje x, ypred - eje y. 
    
    [[172  88  81   6]
    [ 12  65  66   2]
    [  9 272 615  27]
    [  0   0   0   0]]
    
    Parameters
    ----------
    confmat : numpy.ndarray
        Array with n rows, each of one being a flattened confusion matrix.
    index : INT, optional
        Integer for index of the dataframe. The default is 0.
    multiple : BOOL, optional
        If True, returns metrics per CV fold. If False, returns mean and std
        of the metric over all folds (in complex format).

    Returns
    -------
    performance : pd.DataFrame
        Dataframe with all classification performance metrics.
        Use "{0.real:.3} [{0.imag:.2}]".format to display float_format in latex
        
        Example for latex tables:
            print(structConfMat(confmat,multiple=False)
            .to_latex(float_format="{0.real:.3} [{0.imag:.2}]".format))
            
        Note: for coonverting multiple performance to average/std use
            (performance.mean() + 1j*performance.std()).to_frame().T
    """
    
    intdim = int(np.sqrt(confmat.shape[1]))
    conf_n = confmat.reshape((len(confmat), intdim, intdim))
    corrects = conf_n.transpose(2,1,0).reshape((-1,len(conf_n)))[::(intdim+1)]
    corrects = corrects.sum(axis=0)
    n_folds = conf_n.sum(axis=1).sum(axis=1)
    cr = corrects/n_folds
    
    aux_n = conf_n[:,0][:,0]/conf_n[:,0].sum(axis=1)
    for ix in range(intdim-1):
        aux_n = np.c_[aux_n, conf_n[:,ix+1][:,ix+1]/conf_n[:,ix+1].sum(axis=1)]
        
    b_acc = np.nanmean(aux_n, axis=1)
        
    performance = pd.DataFrame({'CorrectRate': cr, 'ErrorRate': 1-cr,
                                'balAcc': b_acc}, 
                               index=index+np.arange(confmat.shape[0]))
    for ix in range(aux_n.shape[1]):
        auxperf = pd.DataFrame({f'Class_{ix}': aux_n[:,ix]}, 
                               index=index+np.arange(confmat.shape[0]))
        performance = pd.concat((performance, auxperf),axis=1)
        
    if intdim==2:
        columns = performance.columns.tolist()
        columns[columns.index('Class_0')]='Specificity'
        columns[columns.index('Class_1')]='Sensitivity'
        performance.columns = columns
        prec = aux_n[:,1]/(aux_n[:,1]+1-aux_n[:,0])
        f1 = 2*prec*aux_n[:,1]/(prec+aux_n[:,1])
        performance['Precision'] = prec
        performance['F1'] = f1

        
    if multiple==False:
        if useratio:
            performance['ratio'] = confmat.sum(axis=1)/confmat.sum()
            performance = (performance.T*performance['ratio']).T.sum()
        else:
            performance = (performance.mean(skipna=True) 
                        + 1j*performance.std(skipna=True)).to_frame().T
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

def test_structconfmat():
    """test structconfmat
    sklearn confusion_matrix gives:
                y_pred
    y_true  [[tn, fp],
             [fn, tp]]
    """
    change=3
    y_true = [[0]*3+[1]*5,
              [0]*4+[1]*4,
              [0]*3+[1]*3]
    y_pred = [el[:] for el in y_true]
    cmat = []
    for ix in range(len(y_pred)):
        y_pred[ix][change] = (y_pred[ix][change]+1)%2
        cmat.append(confusion_matrix(y_true[ix], y_pred[ix]).reshape((-1)))
    cmat = np.array(cmat)
    tn, fp, fn, tp = cmat.sum(axis=0)
    df = structConfMat(cmat)
    assert df['Specificity'].values[0].real==tn/(tn+fp)
    assert df['Sensitivity'].values[0].real==tp/(tp+fn)

def test_multiclass():
    pass

if __name__ == '__main__':
    test_structconfmat()