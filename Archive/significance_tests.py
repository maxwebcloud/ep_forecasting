import scipy.stats as stats

def friedmann_test(rmses1, rmses2, rmses3, rmses4):
    stat, pValue = stats.friedmanchisquare(rmses1, rmses2, rmses3, rmses4)
    return pValue


def wilcoxon_test(rmses1, rmses2):
    stat, pValue = stats.wilcoxon(rmses1, rmses2)
    return pValue

rmsesRNN=[0.0299, 0.0233, 0.0201, 0.0208, 0.0209, 0.0216]
rmsesLSTM=[0.0204, 0.0214, 0.0204, 0.223, 0.0222, 0.0213]
rmsesSTLST=[0.0219, 0.0220, 0.0237, 0.0226, 0.0212, 0.0223]
rmsesPLSTM=[0.0204, 0.0200, 0.0215, 0.0202, 0.0207, 0.0205]

pValueFried = friedmann_test(rmsesRNN, rmsesLSTM, rmsesSTLST, rmsesPLSTM)
print(f"Friedmann-Test: {pValueFried} (p-Value)")

pValueWil = wilcoxon_test(rmsesLSTM, rmsesPLSTM)
print(f"Wilcoxon-Test für LSTM und PLSTM: {pValueWil} (p-Value)")