from Include.Gantt1 import my_gantt
from Include.Individual import GeneticAlgorithm


class MyGeneticAlgorithm(GeneticAlgorithm):

    def run(self):
        self.initialize_connect_class()
        self.get_boys()
        self.get_fitness()
        for _ in range(self.generate_number):
            print(self.boys.iloc[0, -1])
            if len(self.duplicate_removal(self.boys['name'].values.tolist())) < 2:
                break
            self.selection()
            self.crossover()
            self.get_fitness()
        self.marshaling_result()
        my_gantt()
