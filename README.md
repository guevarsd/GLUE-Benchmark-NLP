# Final Project: GLUE Benchmark


The General Language Understanding Evaluation benchmark (GLUE) is a compilation of natural language datasets and tasks designed with the goal of testing models on a variety of different language challenges. 

Historically, many NLP models were trained and tested on very specific datasets. This method often produced well performing models when run on similar data, but had very poor performance when applied to corpuses that varied from their original training data. Additionally, these models could only be used to perform the specific task they were built for, with very little cross-utility to different tasks, even when using the same data.

With the advent of transfer learning, it became possible to create a model which could be used for multiple tasks, across a variety of input datasets. Through this, models that have a deeper understanding of language can be created, with broad cross-utility and applicability in a variety of natural language contexts.

The 9 tasks in the GLUE dataset represent a diverse set of challenges, from grammatical parsing to context-reliant pronoun identification, from an array of text contexts, including news reports, public forums and wikipedia pages. 

To create a high-performing model capable of understanding and generating natural language, we integrate three state-of-the-art models -- Electra, XLNet, and DeBERTa -- together with an ensemble learning model to capture information from the three disparate models on each task. With this method, we capitalize on the unique strengths of each individual model, while mitigating most weaknesses. 


**In this repository**:
The Code folder holds the codes used to implement the ensemble methods and the Results folder contains the predictions of the ensembles.

The DeBERTa and Electra folders on the main branch contain the config and json files of the best model for each task for each Transformer. XLNet's config and json files can be found [here](https://drive.google.com/drive/folders/1r8wa6eLtjMt4jgv96zea-9oFWR8BMtcF?usp=sharing).

##

### The Datasets 
To load and process each of the GLUE tasks we employ the ‘datasets’ package provided by Hugging Face, thus the GLUE tasks are conveniently accessible using the load_dataset() function. The loaded datasets are pre-separated into training, testing and validation datasets.
