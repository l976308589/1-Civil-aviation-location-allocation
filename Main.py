from Code.GeneticAlgorithm import MyGeneticAlgorithm

if __name__ == '__main__':
    MyGeneticAlgorithm(dna_length=50,
                       population_number=100,
                       cross_probability=0.3,
                       mutation_probability=0.06,
                       generate_number=100,
                       target='min').run()

