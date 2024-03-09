### People Name Classifier


Your task is to submit code that allows us to determine whether a given string is the name of a person or not. Here are some expectations: 

* “John Johnson” -> True 
* “Kahfsgjgfjsdgjhfgdjs” -> False 
* “Book” -> False 
* “Jean Claude von Muchausen Gordon-Smith” -> True 

In order to do this, we prepared a toy dataset provided by the DBpedia Association at http://wiki.dbpedia.org/downloads-2016-10#p10608-2. The datasets contain RDF triples that represent facts about the world in the form: subject - predicate - object (e.g. “John loves Mary”). We prepared the dataset along the following way:

1. We extracted the names everything in the knowledge base from the [mappings-based literals file](http://downloads.dbpedia.org/2016-10/core-i18n/en/mappingbased_literals_wkd_uris_en.ttl.bz2) (meaning that the object of the triple is a string literal) and stored into a file (`grep 'foaf/0.1/name' mappingbased_literals_wkd_uris_en.ttl > name.ttl`)

2. We extracted all the entities that are people from the [transitive-type file](http://downloads.dbpedia.org/2016-10/core-i18n/en/instance_types_transitive_wkd_uris_en.ttl.bz2) (meaning the object of the tripple is a type, for example `Person`) and stored into a file (`grep 'ontology/Person' instance_types_transitive_wkd_uris_en.ttl > person.ttl`) 

Therefore the `name.ttl` dataset would contain lines, such as:

```
<http://wikidata.dbpedia.org/resource/Q15703184> <http://xmlns.com/foaf/0.1/name> "19th National Television Awards"@en .
<http://wikidata.dbpedia.org/resource/Q15703194> <http://xmlns.com/foaf/0.1/name> "José Miguel Ruiz Cortés"@en .
<http://wikidata.dbpedia.org/resource/Q15703194> <http://xmlns.com/foaf/0.1/name> "José Ruiz"@en .
<http://wikidata.dbpedia.org/resource/Q15703238> <http://xmlns.com/foaf/0.1/name> "Rainbow Rowell"@en .
<http://wikidata.dbpedia.org/resource/Q15703254> <http://xmlns.com/foaf/0.1/name> "Escape Velocity"@en .
<http://wikidata.dbpedia.org/resource/Q15703257> <http://xmlns.com/foaf/0.1/name> "The Oath"@en .
<http://wikidata.dbpedia.org/resource/Q15703260> <http://xmlns.com/foaf/0.1/name> "Islanded in a Stream of Stars"@en .
<http://wikidata.dbpedia.org/resource/Q15703263> <http://xmlns.com/foaf/0.1/name> "Occupation"@en .
<http://wikidata.dbpedia.org/resource/Q15703266> <http://xmlns.com/foaf/0.1/name> "Precipice"@en .
<http://wikidata.dbpedia.org/resource/Q15703272> <http://xmlns.com/foaf/0.1/name> "Collaborators"@en .
<http://wikidata.dbpedia.org/resource/Q15703276> <http://xmlns.com/foaf/0.1/name> "A Measure of Salvation"@en .
<http://wikidata.dbpedia.org/resource/Q15703279> <http://xmlns.com/foaf/0.1/name> "Hero"@en .
<http://wikidata.dbpedia.org/resource/Q15703281> <http://xmlns.com/foaf/0.1/name> "Unfinished Business"@en .
<http://wikidata.dbpedia.org/resource/Q15703286> <http://xmlns.com/foaf/0.1/name> "The Passage"@en .
<http://wikidata.dbpedia.org/resource/Q15703291> <http://xmlns.com/foaf/0.1/name> "Rapture"@en .
<http://wikidata.dbpedia.org/resource/Q15703294> <http://xmlns.com/foaf/0.1/name> "The Woman King"@en .
<http://wikidata.dbpedia.org/resource/Q15703298> <http://xmlns.com/foaf/0.1/name> "A Day in the Life"@en .

```

and the `person.ttl` dataset would contain lines, like

```
<http://wikidata.dbpedia.org/resource/Q1000005> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000051> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q100005> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000061> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000085> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000203> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000235> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q10002689> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q1000296> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
<http://wikidata.dbpedia.org/resource/Q100030> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://wikidata.dbpedia.org/ontology/Person> .
```

With these two files you are able to generate a dataset containing tuples of the form `(<name>, <class>)` where `<class>` would be either `1` if the subject in the `names.ttl` file is present in the `person.ttl` file or `0` otherwise (binary classification problem).

We have defined a helper `name_classifier.py` file that would help you. It has some boiler plate defined but you need to fill out yourself and write some extra functions or two.

If you use extra libraries, please amend the provided `requirements.txt` file and this readme with instructions. Once everything is ready, we could use the tool by running:

`python name_classifier.py --in-folder <path-to-data> --out-folder <path-to-model-destination>` , where
	* `<path-to-data>` corresponds to the data containing the data files
	* `<path-to-model-destination>` corresponds to a folder where the trained model will be serialised to.


Your code in `name_classifier.py` should:
* generate a training / eval / test split (and do any necessary data pre-processing)
* train a model
* print evaluation metrics
* save it to a destination

Once you are done, submit a pull request for evaluation. The data files are avaliable in zip files in the data folder. Please do not add the extracted data files in the pull request. 


My observations :

Built a character level RNN model. Distinguishing person names from non-person names, because character-level models excel in understanding and generating textual patterns at the most granular level.
Personal names, especially, may include special spellings and hyphenations that are impossible to capture with word models as they depend on their vocabulary set beforehand. Nonetheless, the process of using pretrained models becomes tricky because while these models could be very powerful for capturing semantics at the word level, it may not be applicable here due to the fact that they have been trained using language corpus of general domain which does not reflect subtle patterns and usage specific to personal names. In this case though, taking another approach at a character level allows one to learn directly from the sequence of characters themselves leading to a better comprehension of patterns and sequences which can indicate personal names; thus providing a far more specialized solution for this classification task.

Additional Functions purpose:
1.  The is imbalanced as with : 1895792 - label (0) and 706970 - label (1).
 *  Hence, created a new function `resample_dataset` : to balance the classes by undersampling.
 *  `label (0)` - 706970 
 *  `label (1)` - 706970
2.  create a separate sample to use for prediction later on. This is not part if training. `sample_for_prediction` is used for this purpose.
3.  `clean_name` : Function to clean the name column. But retained apostrophe and diacritics which are important to identify names.
4.  `tokenize_and_pad` : Function is used for tokenizing and padding. Here I have used 95th percentile to cover for the max_Sequence length of names.
5.  `save_tokenizer_and_max_seq_length` and `save_predict_samples` : to save the tokenizer, max_seq_length and a sample input that will be used for prediction.
6.  `consolidated_load_and_predict` : This function will load model and model parameter along with sample input and then predict on this input and results are saved.

`consolidated_load_and_predict` function will be commented in the `name_classifier.py` file.
The results are already saved in prediction directory. In order to execute this function, please uncomment this function and comment out the `train` function.
The function will import all the model parameters and sample input from the model and prediction directory and provide the results as csv in the same directory.

If not running through command line please uncomment these lines respectively.
* line 296 : `train('data','./')` to execute `train` function.
or
* line 342 : `consolidated_load_and_predict('./')` to execute this function.

To execute from command line :
* use this command on terminal : `python name_classifier.py 'data' './'` 

Importance of Prediction on Unseen data:

Making predictions on totally unseen data distinct from both the training and test sets gives a sense of how good a model is at generalizing. In real life scenarios, this is mirrored by the models experiencing completely new kinds of data that were never used for training or validation during testing. It provides useful information on how well the model can recognize patterns that were taught to it when given different inputs whose nature was not anticipated emphasizing its adaptability and dependability beyond any set boundaries. Moreover, it brings out hidden bias or overfitting which might be missed in normal evaluation stages.
Please refer to `prediction_results.csv file` in prediction directory.

Results:

* `Test Accuracy`: 0.810
* `Precision`: 0.77
* `Recall`: 0.87

These outcomes show that the binary classifier for names of persons works properly, with a test accuracy of about 81.10%, which means it reliably recognizes names as people’s names or not. With around 77.29% precision, this suggests that it correctly guesses the name is a person’s name nearly 77% of the time. About 87.7% recall implies that approximately 88% actual personal names in the test data are identified by it.

What's Next : 

To perform better in the future, incorporating more LSTM layers, GRUs or attention mechanisms is a viable option. While considering these possibilities, we need to be careful about overfitting to ensure that the changes we make consistently increase the model’s accuracy and generalizability.