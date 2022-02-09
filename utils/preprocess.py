import pandas as pd
from tqdm import tqdm

from utils.spell import LogParser

dfs = None
tqdm.pandas(desc='classify blk')


def get_blockid(items):
    for item in items.replace('\'', ' ').split(' '):
        if 'blk' in item:
            return item


def getLogKeySeq(BlockId):
    global dfs
    tmpdf = dfs.get_group(BlockId)
    LogKeySeq = tmpdf['EventId'].values
    LogKeySeq = ' '.join(str(i) for i in LogKeySeq)
    return LogKeySeq


def preprocess(generate, train_num=100000, val_split=0.3):
    if generate:
        log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>'
        parser = LogParser(indir='./data/HDFS/', outdir='./data/HDFS/', log_format=log_format)
        parser.parse('HDFS.log')

        df = pd.read_csv('./data/HDFS/HDFS.log_templates.csv')
        EventId2num = {}
        for num, EventId in enumerate(df['EventId'].values):
            EventId2num[EventId] = num

        df = pd.read_csv('./data/HDFS/HDFS.log_structured.csv')
        df['BlockId'] = df['ParameterList'].apply(get_blockid)
        df['EventId'] = df['EventId'].apply(lambda x: EventId2num[x])
        df = df[['LineId', 'EventId', 'BlockId']]

        global dfs
        dfs = df.groupby('BlockId')
        label2id = {'Anomaly': 1, 'Normal': 0}

        label = pd.read_csv('./data/HDFS/anomaly_label.csv')
        label['LogKeySeq'] = label['BlockId'].progress_apply(getLogKeySeq)
        label['Label'] = label['Label'].apply(lambda x: label2id[x])
        label[['BlockId', 'Label', 'LogKeySeq']].to_csv('./data/HDFS/HDFS.csv', index=False)

    df = pd.read_csv('./data/HDFS/HDFS.csv')
    train = df[df['Label'] == 0].sample(n=train_num, replace=False)
    test = df[~df.index.isin(train.index)]

    valid = train.sample(n=int(train_num * val_split), replace=False)
    train = train[~train.index.isin(valid.index)]

    return train, valid, test
