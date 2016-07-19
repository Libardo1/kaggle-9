import pandas as pd


def kaggle_frame(atlas_data):
    """
    A Kaggle submission for ATLAS experimental data is a csv file of the form
    -------------
    EventId,RankOrder,Class
    e_0,r_0,c_0
    e_1,r_1,c_1
    ...
    e_i,r_i,c_i
    ...
    e_n,r_n,c_n
    -------------
    where i is the collision event id, r_i is the rank of event e_i, that is, ordered from
    least signal-like (most background-like) to most signal-like (least background-like),
    r_i is the r_i'th most signal-like event in the data test set. Finally, c_i is the class
    of event e_i generated by the classifier ('signal', or 'background').
    """
    event_ids    = atlas_data.index
    rank_orders  = atlas_data['RankOrder']
    classes      = atlas_data['Label']
    cols         = ['EventId', 'RankOrder', 'Class']
    kaggle_frame = pd.DataFrame({'EventId': event_ids, 'RankOrder': rank_orders, 'Class': classes})

    return kaggle_frame