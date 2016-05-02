# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import h5py
from datetime import datetime, timedelta
import numpy as np
#import scipy as sp
#import statsmodels.api as sm
#import statsmodels.tsa.arima_model as arima


def convert_to_date(num):
    return datetime.strptime(str(int(num)), '%Y%m%d%H%M%S%f') - timedelta(hours=5)


class Plot:

    def __init__(self, group='EURUSD/2015/12', filename='data/forex_data.hdf5'):
        self.dataset_list = []
#        self.volume_list = []
        self.signal = []
        self.signal_const = []
        self.signal_tstamps = []
        self.signal_tstamps_const = []
#        self.volume_data = []
#        self.volume_tstamps = []
        self.plt = plt
        self.group = group
        self.filename = filename

    def iter_datasets(self, name, obj):
        try:
            self.dataset_list.append(np.array(obj))
        except ValueError as e:
            print e
        print name

    def collect_signal(self, r=1):
        with h5py.File(self.filename, "r") as f:
            f[self.group].visititems(self.iter_datasets)
        temp = np.concatenate(self.dataset_list)
        self.signal = temp[0:int(len(temp)*r)]
        self.signal_tstamps = map(convert_to_date, self.signal[:, 0])

    def collect_signal_const(self, r=1, interval=10):
        if not self.signal.any():
            self.collect_signal(r)

        start = self.signal_tstamps[0]
        end =  self.signal_tstamps[-1]

        current = end - timedelta(microseconds = end.microsecond)
        i = -1
        while current > (start + timedelta(seconds=interval)):
            if current.weekday() in [5, 6]:

                current -= timedelta(seconds=interval)
                continue
            self.signal_tstamps_const.append(current)
            signal_bin = []
            while self.signal_tstamps[i] >= (current - timedelta(seconds=interval)):
                signal_bin.append(self.signal[i,1])
                i -= 1
            if signal_bin:
                self.signal_const.append([np.mean(signal_bin),
                                          signal_bin[-1],
                                          signal_bin[0],
                                          min(signal_bin),
                                          max(signal_bin),
                                          len(signal_bin)])
            else:
                self.signal_const.append([self.signal[i, 1],
                                          self.signal[i, 1],
                                          self.signal[i, 1],
                                          self.signal[i, 1],
                                          self.signal[i, 1],
                                          0])
            current -= timedelta(seconds=interval)
        self.signal_const = np.array(self.signal_const)


    def plot_signal(self, r=1):
        self.collect_signal(r)
        self.plt.plot(self.signal_tstamps, self.signal[:,1], 'bo', markersize = 3)

    def plot_signal_const(self, r=1, interval=10):
        if not self.signal_const:
            self.collect_signal_const(r, interval)

        self.plt.plot(self.signal_const[:, 0], 'b.')
#        self.plt.plot(self.signal_tstamps_const, self.signal_const[:, 3], 'r.')
#        self.plt.plot(self.signal_tstamps_const, self.signal_const[:, 4], 'r.')

    def plot_weekends(self):
        date = self.signal_tstamps[0].date()
        end_date = self.signal_tstamps[-1].date()
        weekend_days = []
        while date <= end_date:
            if date.weekday() in [5]:
                weekend_days.append(date)
                print date
                self.plt.axvspan(datetime(date.year, date.month,date.day),
                            datetime(date.year, date.month,date.day) + timedelta(days=2),
                            alpha=0.5,
                            color='r')
            date += timedelta(days=1)

    def plot_ema(self, n, p, linestyle):
        temp = []
        for i in range(n):
            temp.append(i**p)
        mask = list(reversed([float(x)/sum(temp) for x in temp]))

        self.ema = np.convolve(self.signal[:,1], mask, mode='valid')
        print len(self.ema)
        print len(self.tstamps)
        self.plt.plot(self.tstamps[n-1:], self.ema, linestyle, linewidth=1.7)

    def plot_fft(self):
        print 'a'
        signal_fft = np.fft.fft(self.signal_const[:, 0])
        print 'b'
        print signal_fft
        print len(signal_fft)
        print len(self.signal_const[:, 0])
#        print self.signal_const[:,0].shape[-1]
#        freq = np.fft.fftfreq(len(self.signal_const))
#        print freq
#        print signal_fft.real
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.linspace(0,1,len(signal_fft)), signal_fft.real)



p = Plot()
#p.collect_signal_const(r=0.01, interval=10)

p.collect_signal(r=0.2)
p.plot_signal(r=0.2)
#p.plot_signal_const(r=0.3, interval=60)
#p.plot_fft()
#p.plot_weekends()
#p.plot_ema(300, 0, 'r-')
#p.plot_ema(300, 1, 'c-')
#p.plot_ema(300, 2, 'm-')
#p.plot_ema(300, 3, 'k-')
#p.plot_ema(300, 4, 'g-')
#p.plot_ema(300, 5, 'y-')
#p.plot_dft()

