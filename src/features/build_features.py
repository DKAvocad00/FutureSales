import pandas as pd
import itertools
import numpy as np


def create_final_data(train: pd.DataFrame, test: pd.DataFrame | None = None, make_big: bool = False) -> pd.DataFrame:
    train["date_block_num"] = train["date_block_num"].astype(np.int8)
    train["shop_id"] = train['shop_id'].astype(np.int8)
    train["item_id"] = train['item_id'].astype(np.int16)

    df = train.groupby(["date_block_num", "shop_id", "item_id"], as_index=False).agg(
        item_cnt_month=pd.NamedAgg(column="item_cnt_day", aggfunc="sum"),
        item_revenue_month=pd.NamedAgg(column="revenue", aggfunc="sum"),
    )

    if make_big:
        indexlist: list = []
        for i in train['date_block_num'].unique():
            x = itertools.product([i],
                                  train.loc[train['date_block_num'] == i]['shop_id'].unique(),
                                  train.loc[train['date_block_num'] == i]['item_id'].unique(),
                                  )
            indexlist.append(np.array(list(x)))

        sales_big: pd.DataFrame = pd.DataFrame(data=np.concatenate(indexlist, axis=0),
                                               columns=['date_block_num', 'shop_id', 'item_id'])

        df = sales_big.merge(df, how="left", on=['date_block_num', 'shop_id', 'item_id'])

    if test is not None:
        test["date_block_num"] = 34
        test["date_block_num"] = test["date_block_num"].astype(np.int8)
        test["shop_id"] = test['shop_id'].astype(np.int8)
        test["item_id"] = test['item_id'].astype(np.int16)
        test = test.drop(columns="ID")

        df = pd.concat([df, test[["date_block_num", "shop_id", "item_id"]]])

    # df['item_cnt_month'] = df['item_id'].fillna(0)
    # df['item_revenue_month'] = df['item_revenue_month'].fillna(0)

    return df
