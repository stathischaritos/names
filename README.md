Intro
=====
This is a simple classifier for recognizing names of People from random text.
I try four different classification methods from Scikit and also provide an
API to try out or use as a service.

Usage
=====
The project contains three scripts

  - `data.py` for downloading and forming the data set
  - `main.py` for training the model
  - `api.py` for serving the model as a web service.

Other than that I have also added the data.csv file, which can also be
downloaded from DBPedia using the data.py script, but it was IMHO small enough
(33mb) to put up on github for other people to use as is.

Environment
-----------
To run the project for the first time, using python 2.7, run the following
commands:

    pip install virtualenv

    cd project folder

    virtualenv venv

    source venv/bin/activate

    pip install -r requirements.txt

    mkdir model // For storing the models

After that you can deactivate venv by calling

    deactivate

and re-activate it whenever you want by just running

    source venv/bin/activate


Getting the data
----------------
I have uploaded the data.csv file that has the name data, and random text data
comes from the newsgroups of sklearn. The name data was downloaded from DBPedia
using the data.py script.

    python data.py

Training the model
------------------
To train the model run

    python main.py

This trains the NB classifier and saves the model in pickle files. The
confusion matrix and accuracy score is printed in the terminal. You can
uncomment more models inside the code to add them to the process.

Running the api
---------------
I have also added a small flask api to try out the model with user inserted
strings.You can start it using:

    python api.py

Then you can test it at http://127.0.0.1:5000/classify using simple get
requests from a browser or a tool like Postman. Give it a query in the url
query parameters and it will return the probability that the string you
provided is the name of a person or not.

  ex.

    http://127.0.0.1:5000/classify?q="John Smith"
    result: {"multi_nb": [0.020987041619013835, 0.9790129583809848]}

    http://127.0.0.1:5000/classify?q="Hello World"
    {"multi_nb": [0.9917522714060827, 0.008247728593917876]}


Issues:
=======

  - Currently only the random forest classifier seems to be able to correctly
  classify gibberish (ex. "fdsafdsafdsafdsafdas"). I tried character n-grams
  and word n-grams. The character n-grams perform worse than random. Word
  n-gram do not help. One solution could be to introduce completely random text
  in the training set. Another could be to label non dictionary words as negative.
