import numpy as np
import talib
import requests
from plotly.offline import plot
import plotly.graph_objs as go
import pandas as pd
import time
import datetime
from datetime import date
from itertools import compress

def candleGetNobitex(count,Symbol,frame):
    candle=todayOneHundred(count,Symbol,frame)
    df = pd.DataFrame(candle,
                             columns=['c', 'h', 'l', 'o', 't','v'],
                             dtype='float64')
    posix_time = pd.to_datetime(df['t'], unit='s')
# append posix_time
    df.insert(0, "time", posix_time)
# drop unix time stamp
    df.drop("t", axis = 1, inplace = True)
    #print(df)
    candle_names = talib.get_function_groups()['Pattern Recognition']
    #print(candle_names)
    op = df['o']
    hi = df['h']
    lo = df['l']
    cl = df['c']
    op = np.array(op, dtype=float)
    hi = np.array(hi, dtype=float)
    lo = np.array(lo, dtype=float)
    cl = np.array(cl, dtype=float)
# create columns for each pattern
    for candle in candle_names:
             df[candle] = getattr(talib, candle)(op, hi, lo, cl)
    recognize_candlestick(df)
    #print(df)
    return (df)

candle_rankings = {
        "CDL3LINESTRIKE_Bull": 1,
        "CDL3LINESTRIKE_Bear": 2,
        "CDL3BLACKCROWS_Bull": 3,
        "CDL3BLACKCROWS_Bear": 3,
        "CDLEVENINGSTAR_Bull": 4,
        "CDLEVENINGSTAR_Bear": 4,
        "CDLTASUKIGAP_Bull": 5,
        "CDLTASUKIGAP_Bear": 5,
        "CDLINVERTEDHAMMER_Bull": 6,
        "CDLINVERTEDHAMMER_Bear": 6,
        "CDLMATCHINGLOW_Bull": 7,
        "CDLMATCHINGLOW_Bear": 7,
        "CDLABANDONEDBABY_Bull": 8,
        "CDLABANDONEDBABY_Bear": 8,
        "CDLBREAKAWAY_Bull": 10,
        "CDLBREAKAWAY_Bear": 10,
        "CDLMORNINGSTAR_Bull": 12,
        "CDLMORNINGSTAR_Bear": 12,
        "CDLPIERCING_Bull": 13,
        "CDLPIERCING_Bear": 13,
        "CDLSTICKSANDWICH_Bull": 14,
        "CDLSTICKSANDWICH_Bear": 14,
        "CDLTHRUSTING_Bull": 15,
        "CDLTHRUSTING_Bear": 15,
        "CDLINNECK_Bull": 17,
        "CDLINNECK_Bear": 17,
        "CDL3INSIDE_Bull": 20,
        "CDL3INSIDE_Bear": 56,
        "CDLHOMINGPIGEON_Bull": 21,
        "CDLHOMINGPIGEON_Bear": 21,
        "CDLDARKCLOUDCOVER_Bull": 22,
        "CDLDARKCLOUDCOVER_Bear": 22,
        "CDLIDENTICAL3CROWS_Bull": 24,
        "CDLIDENTICAL3CROWS_Bear": 24,
        "CDLMORNINGDOJISTAR_Bull": 25,
        "CDLMORNINGDOJISTAR_Bear": 25,
        "CDLXSIDEGAP3METHODS_Bull": 27,
        "CDLXSIDEGAP3METHODS_Bear": 26,
        "CDLTRISTAR_Bull": 28,
        "CDLTRISTAR_Bear": 76,
        "CDLGAPSIDESIDEWHITE_Bull": 46,
        "CDLGAPSIDESIDEWHITE_Bear": 29,
        "CDLEVENINGDOJISTAR_Bull": 30,
        "CDLEVENINGDOJISTAR_Bear": 30,
        "CDL3WHITESOLDIERS_Bull": 32,
        "CDL3WHITESOLDIERS_Bear": 32,
        "CDLONNECK_Bull": 33,
        "CDLONNECK_Bear": 33,
        "CDL3OUTSIDE_Bull": 34,
        "CDL3OUTSIDE_Bear": 39,
        "CDLRICKSHAWMAN_Bull": 35,
        "CDLRICKSHAWMAN_Bear": 35,
        "CDLSEPARATINGLINES_Bull": 36,
        "CDLSEPARATINGLINES_Bear": 40,
        "CDLLONGLEGGEDDOJI_Bull": 37,
        "CDLLONGLEGGEDDOJI_Bear": 37,
        "CDLHARAMI_Bull": 38,
        "CDLHARAMI_Bear": 72,
        "CDLLADDERBOTTOM_Bull": 41,
        "CDLLADDERBOTTOM_Bear": 41,
        "CDLCLOSINGMARUBOZU_Bull": 70,
        "CDLCLOSINGMARUBOZU_Bear": 43,
        "CDLTAKURI_Bull": 47,
        "CDLTAKURI_Bear": 47,
        "CDLDOJISTAR_Bull": 49,
        "CDLDOJISTAR_Bear": 51,
        "CDLHARAMICROSS_Bull": 50,
        "CDLHARAMICROSS_Bear": 80,
        "CDLADVANCEBLOCK_Bull": 54,
        "CDLADVANCEBLOCK_Bear": 54,
        "CDLSHOOTINGSTAR_Bull": 55,
        "CDLSHOOTINGSTAR_Bear": 55,
        "CDLMARUBOZU_Bull": 71,
        "CDLMARUBOZU_Bear": 57,
        "CDLUNIQUE3RIVER_Bull": 60,
        "CDLUNIQUE3RIVER_Bear": 60,
        "CDL2CROWS_Bull": 61,
        "CDL2CROWS_Bear": 61,
        "CDLBELTHOLD_Bull": 62,
        "CDLBELTHOLD_Bear": 63,
        "CDLHAMMER_Bull": 65,
        "CDLHAMMER_Bear": 65,
        "CDLHIGHWAVE_Bull": 67,
        "CDLHIGHWAVE_Bear": 67,
        "CDLSPINNINGTOP_Bull": 69,
        "CDLSPINNINGTOP_Bear": 73,
        "CDLUPSIDEGAP2CROWS_Bull": 74,
        "CDLUPSIDEGAP2CROWS_Bear": 74,
        "CDLGRAVESTONEDOJI_Bull": 77,
        "CDLGRAVESTONEDOJI_Bear": 77,
        "CDLHIKKAKEMOD_Bull": 82,
        "CDLHIKKAKEMOD_Bear": 81,
        "CDLHIKKAKE_Bull": 85,
        "CDLHIKKAKE_Bear": 83,
        "CDLENGULFING_Bull": 84,
        "CDLENGULFING_Bear": 91,
        "CDLMATHOLD_Bull": 86,
        "CDLMATHOLD_Bear": 86,
        "CDLHANGINGMAN_Bull": 87,
        "CDLHANGINGMAN_Bear": 87,
        "CDLRISEFALL3METHODS_Bull": 94,
        "CDLRISEFALL3METHODS_Bear": 89,
        "CDLKICKING_Bull": 96,
        "CDLKICKING_Bear": 102,
        "CDLDRAGONFLYDOJI_Bull": 98,
        "CDLDRAGONFLYDOJI_Bear": 98,
        "CDLCONCEALBABYSWALL_Bull": 101,
        "CDLCONCEALBABYSWALL_Bear": 101,
        "CDL3STARSINSOUTH_Bull": 103,
        "CDL3STARSINSOUTH_Bear": 103,
        "CDLDOJI_Bull": 104,
        "CDLDOJI_Bear": 104
    }

def recognize_candlestick(df):
    """
    Recognizes candlestick patterns and appends 2 additional columns to df;
    1st - Best Performance candlestick pattern matched by www.thepatternsite.com
    2nd - # of matched patterns
    """
    op = df['o'].astype(float)
    hi = df['h'].astype(float)
    lo = df['l'].astype(float)
    cl = df['c'].astype(float)
    op = np.array(op, dtype=float)
    hi = np.array(hi, dtype=float)
    lo = np.array(lo, dtype=float)
    cl = np.array(cl, dtype=float)

    candle_names = talib.get_function_groups()['Pattern Recognition']

    # patterns not found in the patternsite.com
    exclude_items = ('CDLCOUNTERATTACK',
                     'CDLLONGLINE',
                     'CDLSHORTLINE',
                     'CDLSTALLEDPATTERN',
                     'CDLKICKINGBYLENGTH')
#exclude_items = ('CDLCOUNTERATTACK',
 #                    'CDLLONGLINE',
 #                    'CDLSHORTLINE',
  #                   'CDLSTALLEDPATTERN',
   #                  'CDLKICKINGBYLENGTH')

    #candle_names = [candle for candle in candle_names]
    candle_names = [candle for candle in candle_names if candle not in exclude_items]


    # create columns for each candle
    for candle in candle_names:
        # below is same as;
        # df["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(op, hi, lo, cl)
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)


    df['candlestick_pattern'] = np.nan
    df['candlestick_match_count'] = np.nan
    for index, row in df.iterrows():
        df.loc[index, 'candlestick_match_count']=0
        # no pattern found
        if len(row[candle_names]) - sum(row[candle_names] == 0) == 0:
            df.loc[index,'candlestick_pattern'] = "NO Position"
            df.loc[index, 'candlestick_match_count'] = 0
        # single pattern found
        elif len(row[candle_names]) - sum(row[candle_names] == 0) == 1:
            # bull pattern 100 or 200
            if any(row[candle_names].values > 0):
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bull'
                if( candle_rankings[pattern]<10000):
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] =(104- candle_rankings[pattern])+ df.loc[index, 'candlestick_match_count']
                else:
                    df.loc[index,'candlestick_pattern'] = "BAD_PATTERN"
                    df.loc[index, 'candlestick_match_count'] = 0
            # bear pattern -100 or -200
            else:
                pattern = list(compress(row[candle_names].keys(), row[candle_names].values != 0))[0] + '_Bear'
                if( candle_rankings[pattern]<10000):
                    df.loc[index, 'candlestick_pattern'] = pattern
                    df.loc[index, 'candlestick_match_count'] = (104- candle_rankings[pattern])+ df.loc[index, 'candlestick_match_count']
                else:
                    df.loc[index,'candlestick_pattern'] = "BAD_PATTERN"
                    df.loc[index, 'candlestick_match_count'] = 0
        # multiple patterns matched -- select best performance
        else:
            # filter out pattern names from bool list of values
            patterns = list(compress(row[candle_names].keys(), row[candle_names].values != 0))
            container = []
            bear=0
            bull=0
            for pattern in patterns:
                if row[pattern] > 0:
                    container.append(pattern + '_Bull')
                    bull=1
                else:
                    container.append(pattern + '_Bear')
                    bear=1
            rank_list = [candle_rankings[p] for p in container]
            if len(rank_list) == len(container):
                rank_index_best = rank_list.index(min(rank_list))
                if(bull==1 and bear==1):
                    df.loc[index,'candlestick_pattern'] = "No Position"
                    df.loc[index, 'candlestick_match_count'] = 0
                else:
                    if( candle_rankings[container[rank_index_best]]<1000):
                        df.loc[index, 'candlestick_pattern'] = container[rank_index_best]
                        df.loc[index, 'candlestick_match_count'] = len(container)
                    else:
                        df.loc[index,'candlestick_pattern'] = "BAD_PATTERN"
                        df.loc[index, 'candlestick_match_count'] = 0

                
    # clean up candle columns
    cols_to_drop = candle_names + list(exclude_items)
    df.drop(cols_to_drop, axis = 1, inplace = True)

    return 

def todayOneHundred(count,symbol,frame): 
    today= datetime.datetime.now()
    d = datetime.datetime.now()
    unixtimestart = time.mktime(d.timetuple())
    #print(int(unixtimestart))
    url = 'https://api.nobitex.ir/market/udf/history?symbol='+symbol+'&resolution='+frame+'&to='+str(int(unixtimestart))+'&countback='+count
    response = requests.get(url)
    #print(response.json())
    return(response.json())


def plotOneMinutes():
    df = candleGetNobitex('50','BTCUSDT','1')
    #print (df)
    o = df['o'].astype(float)
    h = df['h'].astype(float)
    #print(h)
    l = df['l'].astype(float)
    c = df['c'].astype(float)
    # t= df['candlestick_pattern'].cat(df['candlestick_match_count'])
    t = df[['candlestick_pattern','candlestick_match_count']].apply(lambda x : '{} {}'.format(x[0],x[1]), axis=1)
    trace = go.Candlestick(
                open=o,
                high=h,
                low=l,
                close=c,
                text=t)
    #print(trace)
    data = [trace]

    ADXData = talib.ADX(h.to_numpy(), df['l'].to_numpy(), df['c'].to_numpy(), timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['c'].to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9)
    RSIData = talib.RSI(df['c'].to_numpy(), timeperiod=14)
    print(ADXData)
    ADXtrend= ADXData[len(ADXData)-1]- ADXData[len(ADXData)-2]
    ADXtrendprecent=(ADXtrend*100)/ADXData[len(ADXData)-2]
    Technical="ADX : "+ str(ADXData[len(ADXData)-1])+" ADX Trend : "+str(ADXtrend)+" "+str(ADXtrendprecent)+"% histogram : "+str(macdhist[len(macdhist)-1])+" RSI : "+str(RSIData[len(RSIData)-1])

    layout = {
        'title': Technical,
        'yaxis': {'title': 'Price'},
        'xaxis': {'title': 'Index Number'},
    }
    fig = dict(data=data, layout=layout)
    plot(fig, filename='BTCUSDT1.html',auto_open=False)
    return



def plotFiftyMinutes():
    df = candleGetNobitex('50','BTCUSDT','15')
    #print (df)
    o = df['o'].astype(float)
    h = df['h'].astype(float)
    #print(h)
    l = df['l'].astype(float)
    c = df['c'].astype(float)
    # t= df['candlestick_pattern'].cat(df['candlestick_match_count'])
    t = df[['candlestick_pattern','candlestick_match_count']].apply(lambda x : '{} {}'.format(x[0],x[1]), axis=1)
    trace = go.Candlestick(
                open=o,
                high=h,
                low=l,
                close=c,
                text=t)
    #print(trace)
    data = [trace]

    ADXData = talib.ADX(h.to_numpy(), df['l'].to_numpy(), df['c'].to_numpy(), timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['c'].to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9)
    RSIData = talib.RSI(df['c'].to_numpy(), timeperiod=14)
    print(ADXData)
    ADXtrend= ADXData[len(ADXData)-1]- ADXData[len(ADXData)-2]
    ADXtrendprecent=(ADXtrend*100)/ADXData[len(ADXData)-2]
    Technical="ADX : "+ str(ADXData[len(ADXData)-1])+" ADX Trend : "+str(ADXtrend)+" "+str(ADXtrendprecent)+"% histogram : "+str(macdhist[len(macdhist)-1])+" RSI : "+str(RSIData[len(RSIData)-1])

    layout = {
        'title': Technical,
        'yaxis': {'title': 'Price'},
        'xaxis': {'title': 'Index Number'},
    }
    fig = dict(data=data, layout=layout)
    plot(fig, filename='BTCUSDT15.html',auto_open=False)
    return

def plotFiveMinutes():
    df = candleGetNobitex('50','BTCUSDT','5')
    #print (df)
    o = df['o'].astype(float)
    h = df['h'].astype(float)
    #print(h)
    l = df['l'].astype(float)
    c = df['c'].astype(float)
    # t= df['candlestick_pattern'].cat(df['candlestick_match_count'])
    t = df[['candlestick_pattern','candlestick_match_count']].apply(lambda x : '{} {}'.format(x[0],x[1]), axis=1)
    trace = go.Candlestick(
                open=o,
                high=h,
                low=l,
                close=c,
                text=t)
    #print(trace)
    data = [trace]

    ADXData = talib.ADX(h.to_numpy(), df['l'].to_numpy(), df['c'].to_numpy(), timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['c'].to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9)
    RSIData = talib.RSI(df['c'].to_numpy(), timeperiod=14)
    print(ADXData)
    ADXtrend= ADXData[len(ADXData)-1]- ADXData[len(ADXData)-2]
    ADXtrendprecent=(ADXtrend*100)/ADXData[len(ADXData)-2]
    Technical="ADX : "+ str(ADXData[len(ADXData)-1])+" ADX Trend : "+str(ADXtrend)+" "+str(ADXtrendprecent)+"% histogram : "+str(macdhist[len(macdhist)-1])+" RSI : "+str(RSIData[len(RSIData)-1])

    layout = {
        'title': Technical,
        'yaxis': {'title': 'Price'},
        'xaxis': {'title': 'Index Number'},
    }
    fig = dict(data=data, layout=layout)
    plot(fig, filename='BTCUSDT5.html',auto_open=False)
    return

def plotSixtyMinutes():
    df = candleGetNobitex('50','BTCUSDT','30')
    #print (df)
    o = df['o'].astype(float)
    h = df['h'].astype(float)
    #print(h)
    l = df['l'].astype(float)
    c = df['c'].astype(float)
    # t= df['candlestick_pattern'].cat(df['candlestick_match_count'])
    t = df[['candlestick_pattern','candlestick_match_count']].apply(lambda x : '{} {}'.format(x[0],x[1]), axis=1)
    trace = go.Candlestick(
                open=o,
                high=h,
                low=l,
                close=c,
                text=t)
    #print(trace)
    data = [trace]

    ADXData = talib.ADX(h.to_numpy(), df['l'].to_numpy(), df['c'].to_numpy(), timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['c'].to_numpy(), fastperiod=12, slowperiod=26, signalperiod=9)
    RSIData = talib.RSI(df['c'].to_numpy(), timeperiod=14)
    print(ADXData)
    ADXtrend= ADXData[len(ADXData)-1]- ADXData[len(ADXData)-2]
    ADXtrendprecent=(ADXtrend*100)/ADXData[len(ADXData)-2]
    Technical="ADX : "+ str(ADXData[len(ADXData)-1])+" ADX Trend : "+str(ADXtrend)+" "+str(ADXtrendprecent)+"% histogram : "+str(macdhist[len(macdhist)-1])+" RSI : "+str(RSIData[len(RSIData)-1])

    layout = {
        'title': Technical,
        'yaxis': {'title': 'Price'},
        'xaxis': {'title': 'Index Number'},
    }
    fig = dict(data=data, layout=layout)
    plot(fig, filename='BTCUSDT30.html',auto_open=False)
    return

while 1:
    plotFiftyMinutes()
    plotFiveMinutes()
    plotSixtyMinutes()
    plotOneMinutes()
    time.sleep(40)