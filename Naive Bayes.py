from math import sqrt
from math import exp
from math import pi



def remove_features(lines):
    dataset = []
    for line in lines: 
        line = line.strip().split(",")
        new_line = line[:-4]
        new_line.append(line[57])
        dataset.append(new_line)
    return dataset
        
def separate_by_class(dataset):
    separated = dict()
    separated["1"] = list()
    separated["0"] = list()

    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        vector = list(map(float, vector))
        separated[class_value].append(vector)
    return separated 


def mean(numbers):
	return sum(numbers)/float(len(numbers))
 

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
  

def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries
 

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries  

    
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities







###############################################################################



def main():
    file = open("C:\\Users\\D\\Desktop\\spambase.data")
    dataset = remove_features(file.readlines())
    
    summary = summarize_by_class(dataset)
    for label in summary:
        print(label)
        for row in summary[label]:
            print(row)

    summaries = summarize_by_class(dataset)

   
    line = list(map(float, dataset[0]))
    probabilities = calculate_class_probabilities(summaries, line)
    print(probabilities)

if __name__ == "__main__":
    main()