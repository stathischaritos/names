import sys
import csv
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.datasets import fetch_20newsgroups
from random import randint, shuffle

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

def getAllResourcesPagingQuery (limit, offset):
    return """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT DISTINCT ?name
        WHERE {
            ?person a dbo:Person ;
                      foaf:name ?name .
        }
        LIMIT %i OFFSET %i
    """ % (limit, offset)

def getAllResources (batch=10000, max_resources=0, offset=0):
    print "Retrieving all person names from DBPedia..."
    if max_resources:
        count = max_resources
        limit = min(batch, max_resources);
    else:
        count = 0
        limit = batch
    iterator = offset
    results_left = 1
    while results_left:
        print "Retrieved ", iterator, " names"
        query = getAllResourcesPagingQuery(limit, iterator)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        results = results["results"]["bindings"]
        results_left = len(results)
        for result in results:
            iterator += 1
            if 'name' in result:
                doc = result['name']['value'].encode('utf-8').strip()
                yield [doc]


def genNames(n=0):
    return csv.reader(open('data.csv', 'rb'))

def genRandomNewsgroupsText(n=1000):
    newgroups_data = fetch_20newsgroups(shuffle=True, random_state=42).data
    max_n = len(newgroups_data)
    for i in range(n):
        # Getting a random slice of text, of random size, from a random item of news
        random_item_index = randint(0, max_n - 1)
        random_item = newgroups_data[random_item_index]
        # Arbitrary choice of text to be 10 to 100 characters long
        random_text_length = randint(10, 100)
        idx = randint(0, len(random_item) - random_text_length)
        yield random_item[idx:idx+random_text_length]

def buildTrainingSet(n=10000):
    """
    I'm going to combine random newgroup text and dbpedia names to create a
    balanced label training and test set.
    """
    names = genNames()
    random_text = genRandomNewsgroupsText(n=n)
    for x, y in zip(names, random_text):
        yield  [ x[0], 1 ]
        yield  [ y, 0 ]

def getTrainingSet(n=10000):
    # Consume generator in order to shuffle
    training_set = [ item for item in buildTrainingSet(n=n) ]
    shuffle(training_set)
    data, targets = zip(*training_set)
    return data, targets

if __name__ == "__main__":
    data_file = open('data.csv', 'rb')
    reader = csv.reader(data_file)
    offset = sum(1 for row in reader)
    data_file.close()
    with open('data.csv', "a") as csv_file:
        print "Starting with offset ", offset
        writes = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
        writes.writerows(getAllResources(offset=offset))