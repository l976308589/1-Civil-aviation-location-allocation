import sys
from copy import deepcopy
from functools import reduce
from random import choice

import numpy as np
import pandas as pd


class GeneticAlgorithmConnect(object):
    def __init__(self, min_ini=pd.Timedelta('0 days 00:15:00')):
        self.flight_info = pd.DataFrame()
        self.min_ini = min_ini

    def initialize_flight_info(self):
        signal_day_ = pd.read_csv('Bin\\Data\\Signal_day.csv')
        signal_day_['DES_REAL_LANDING'] = pd.to_datetime(signal_day_['DES_REAL_LANDING'])
        signal_day_['DEP_REAL_TAKEOFF'] = pd.to_datetime(signal_day_['DEP_REAL_TAKEOFF']) + self.min_ini
        self.flight_info = signal_day_.sort_values('DES_REAL_LANDING')

    def marshaling_result(self, seat):
        self.flight_info['Seat'] = seat
        self.flight_info['DEP_REAL_TAKEOFF'] -= self.min_ini
        self.flight_info.to_csv('Bin\\Data\\Result.csv', index=False)


class GeneticAlgorithm:
    def __init__(self,
                 dna_length=50,
                 population_number=100,
                 cross_probability=0.3,
                 mutation_probability=0,
                 generate_number=20,
                 target='max'
                 ):
        self.dna_length = dna_length
        self.population_number = population_number
        self.cross_probability = cross_probability
        self.mutation_probability = mutation_probability
        self.generate_number = generate_number
        self.target = 1 if target == 'max' else -1

        self.connect_data = GeneticAlgorithmConnect()

        self.boys = pd.DataFrame([], columns=['name', 'fitness', 'actual_fitness'])

    def initialize_connect_class(self):
        self.connect_data.initialize_flight_info()

    def decimal_individual(self):
        fi = self.connect_data.flight_info
        pork = list(range(self.dna_length))
        boy = []
        for index, row in fi.iterrows():
            valid = self.valid_pork(boy, row) + list(set(pork) ^ set(list(set(boy)))) if boy else pork
            if valid:
                boy.append(choice(valid))
            else:
                print('初代无解,程序退出')
                sys.exit()
        return boy

    def valid_pork(self, boy, row):
        assigned_position = self.connect_data.flight_info.iloc[:len(boy)].copy()
        assigned_position['Seat'] = boy
        takeoff_time = assigned_position.groupby('Seat').apply(lambda x: x['DEP_REAL_TAKEOFF'].max())
        return takeoff_time[takeoff_time < row['DES_REAL_LANDING']].index.tolist()

    @staticmethod
    def duplicate_removal(some_list):
        return reduce(lambda x, y: x if y in x else x + [y], [[], ] + some_list)

    def get_boys(self):
        boys = []
        while len(boys) < self.population_number:
            boys.append(self.decimal_individual())
            boys = self.duplicate_removal(boys)
        self.boys = pd.DataFrame([], columns=['name', 'fitness'])  # Collecting fitness in a moment
        self.boys['name'] = boys

    def get_fitness(self):
        fi = self.connect_data.flight_info.copy()
        fitness = []
        for boy in self.boys['name'].values:
            fi['Seat'] = boy
            fitness.append(fi.groupby('Seat').apply(self.get_diff).sum())
        self.make_up(fitness)  # never go out without make-up
        self.sort_values()

    def make_up(self, fitness):
        self.boys['actual_fitness'] = fitness
        self.boys['fitness'] = fitness
        self.boys = self.boys.fillna(max(fitness) if self.target > 0 else min(fitness))
        self.boys['fitness'] *= self.target
        self.boys['fitness'] = self.boys['fitness'] - self.boys['fitness'].min() + 1e-5
        self.boys['fitness'] = self.boys['fitness'].ffill()

    def sort_values(self):  # It's always a ascending order
        self.boys = self.boys.sort_values('fitness').reset_index(drop=True)

    @staticmethod
    def get_diff(x):
        x = x.sort_values('DES_REAL_LANDING')
        if (x['DES_REAL_LANDING'] < x['DEP_REAL_TAKEOFF'].shift()).any():
            return np.nan
        return (x['DES_REAL_LANDING'] - x['DEP_REAL_TAKEOFF'].shift()).sum().total_seconds()

    def selection(self):
        idx = np.random.choice(self.population_number, size=self.population_number, replace=True,
                               p=self.boys['fitness'] / self.boys['fitness'].sum())
        self.boys = self.boys.iloc[idx].copy()
        self.sort_values()

    def crossover(self):
        boys = self.boys['name'].values
        girls = deepcopy(boys)

        def should_recovery(index_):
            if not self.is_the_boy_healthy(boy):
                boy[:] = girls[index_]

        for index, boy in enumerate(boys):  # some children is unattractive
            if index == 0:
                boy[:] = girls[-1]
            elif np.random.rand() < self.cross_probability:  # the boy should change
                girl = choice(girls)
                exchange_of_chromosomes = np.random.randint(self.dna_length, size=self.population_number)
                for chromosome in list(set(exchange_of_chromosomes.tolist())):
                    boy[chromosome] = girl[chromosome]
                should_recovery(index)

                if self.mutation_probability > 0:
                    for chromosome in range(self.dna_length):
                        if np.random.rand() < self.mutation_probability:
                            boy[chromosome] = np.random.randint(self.dna_length)
                    should_recovery(index)

    def is_the_boy_healthy(self, boy):
        fi = self.connect_data.flight_info.copy()
        fi['Seat'] = boy
        return fi.groupby('Seat').apply(self.physical_examination).all()

    @staticmethod
    def physical_examination(x):
        if x.shape[0] > 1:
            x = x.sort_values('DES_REAL_LANDING')
            return not (x['DEP_REAL_TAKEOFF'].shift() > x['DES_REAL_LANDING']).any()
        return True

    def marshaling_result(self):
        self.connect_data.marshaling_result(self.boys.iloc[0]['name'])
