import numpy as np
import pandas as pd

def structConfMat(confmat, index=0, multiple=False):
    """
    Creates a pandas dataframe from the confusion matrix. It distinguishes
    between binary and multi-class classification. 

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
        columns[columns.index('Class_0')]='Sensitivity'
        columns[columns.index('Class_1')]='Specificity'
        performance.columns = columns
        prec = aux_n[:,1]/(aux_n[:,1]+1-aux_n[:,0])
        f1 = 2*prec*aux_n[:,1]/(prec+aux_n[:,1])
        performance['Precision'] = prec
        performance['F1'] = f1

        
    if multiple:
        performance = (performance.mean() + 1j*performance.std()).to_frame().T
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
