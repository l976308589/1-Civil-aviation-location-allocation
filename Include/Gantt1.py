# -*- coding: utf-8 -*-

"""Main module."""

from collections import OrderedDict

import matplotlib.dates as m_dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def gantt(task=None, start=None, finish=None, **kwargs):
    """ Plot a gantt chart.
    """

    if 'task_type' in kwargs:
        task_type = kwargs['task_type']
    if 'color' in kwargs:
        color = kwargs['color']

    USES_DATES = False
    if np.issubdtype(start.dtype, np.datetime64):
        start = m_dates.date2num(start)
        USES_DATES = True
    if np.issubdtype(finish.dtype, np.datetime64):
        finish = m_dates.date2num(finish)

    delta = finish - start

    ax = plt.gca()

    labels = []

    # TODO: refactor?    
    encoded_tasks = OrderedDict()
    k = 0
    for n in task:
        if not n in encoded_tasks:
            encoded_tasks[n] = k
            k += 1

    labels = list(encoded_tasks)
    for i, task in enumerate(task):
        j = encoded_tasks[task]
        if color:
            c = color[task_type[i]]
        else:
            c = None
        ax.broken_barh([(start[i], delta[i])], (j - 0.4, 0.8), color=c)

    # Set yticks
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # Set xticks formatting
    # TODO: use matplotlib.dates.AutoDateFormatter
    if USES_DATES:
        ax.xaxis.set_major_formatter(m_dates.DateFormatter('%Y-%m-%d %H:%M'))
        fig = plt.gcf()
        fig.autofmt_xdate()

    ax.invert_yaxis()


def my_gantt(show_pic=False):
    _data = pd.read_csv('Bin\\Data\\Result.csv')[['FLIGHTNO', 'DES_REAL_LANDING', 'DEP_REAL_TAKEOFF', 'Seat']]
    _data['DES_REAL_LANDING'] = pd.to_datetime(_data['DES_REAL_LANDING'])
    _data['DEP_REAL_TAKEOFF'] = pd.to_datetime(_data['DEP_REAL_TAKEOFF'])
    _data['Type'] = "main"
    _data.iloc[1::2, -1] = "sub"
    _data = _data.sort_values('Seat')
    fig = plt.figure(figsize=(30, 20))
    gantt(task=_data['Seat'], start=_data['DES_REAL_LANDING'],
          finish=_data['DEP_REAL_TAKEOFF'], task_type=_data['Type'],
          color={"main": "steelblue", "sub": "lightgreen"})
    plt.title('Gantt', {'fontsize': 14, 'fontweight': 'heavy'})
    if show_pic:
        plt.show()
    fig.savefig('Bin\\Data\\Gantt.png')
