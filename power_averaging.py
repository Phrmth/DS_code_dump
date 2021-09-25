# this is used to take the average of correlated mdoels 

import pandas as pd
import numpy as np

pred1 = pd.read_csv("../input/sep-25/subs_catboost.csv")
pred2 = pd.read_csv("../input/outputs/catboost_5fold (1).csv")
pred3 = pd.read_csv("../input/sep-25/subs_lgb.csv")
pred4 = pd.read_csv("../input/output2/catboost_classifier.csv")
pred5 = pd.read_csv("../input/tps9-81826/submission.csv")
pred6 = pd.read_csv("../input/sep25/subs_catboost_SC.csv")


import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

hist_data = [ pred1.claim, pred2.claim, pred3.claim, pred4.claim, pred5.claim, pred6.claim]

group_labels = ['cat1','cat2','lgb3','cat4','lgb5','cat6']
fig = ff.create_distplot(hist_data, group_labels, bin_size = .3, show_hist = False, show_rug = False)
fig.show()


ensemble = pred1.copy()
ensemble.loc[:,'claim'] = ( pred1.claim**4 + pred2.claim**4 + pred3.claim**4 + pred4.claim**4 + pred5.claim**4 + pred6.claim**4)/ 6


ensemble.to_csv("submission.csv", index = False)
