# happybirds

This is a repository for a university project. The task is to create a machine learning model to predict the sentiment of a tweet about airlines based on annotated data. We use the readme for brainstorming.

---------------------------------------------------------------------------------------------------------------------------------


ideas:

FE extraction from RAW
* UPPERCASE
* emojis
* punctuation signs (!;?;...;..)
* text length
* Timestamp

CountVectorizer bigrams y trigrams
* Ignoring case
* Ignoring punctuation
* stopwrods
* Lematizción

Comments from class 01/02/2018:
* We can use the labels to obtain more suitable vocabulary, maybe neutral comments are misleading.
* We need to do crossvalidation at some point
* The order of words can change sentiment
* For next class (17/04/2018) we have to upload to kaggle results with a most suited model
* We will need to use feature selection [we will see it in next classes]
* Naive Bayes only considers if the word is in the text, it does not consider number of appearences
* At the end of the project (03/07/2018), we will need to present a news repport talking about some airline companies

Comments from class 24/05/2018:

* **15 de June** is the deadline to present boths kaggles!

* Tenemos que hacer una web/blog con una explicación técnica de lo que hemos hecho --> 3 Julio. Memoria. Que sea didactico.
Link al github con el codigo si es posible.

  * Datacleaning
  * exploración de datos
  * machine learning...

  * El metodo que hemos hecho para presentar el kaggle

  * Además de esto ver qué historias podemos extraer sobre las aerolineas.

* **3 Julio** haremos la presentación con un ppt o similar. 2-3 min en el programa que hemos hecho + 7 min con las preguntas/conclusiones y preguntas que nos hemos hecho.


Models
* Naive bayes
* SVM-RBF
* RF

* Boosting


Visualization
* distribution 

How to handle the unevenly distributed dataset? See percents:  
* negative    0.629668  
* neutral     0.210155  
* positive    0.160178  
