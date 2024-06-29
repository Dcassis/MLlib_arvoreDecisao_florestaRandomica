<h1>MLlib: Árvore de decisão e floresta randômica</h1>

Projeto de Classificação com Spark ML
Este repositório contém um exemplo de aplicação de algoritmos de classificação usando o Apache Spark ML.

Descrição
O projeto utiliza o Spark para treinar e avaliar modelos de classificação usando os conjuntos de dados Iris. Dois modelos foram implementados e avaliados: Decision Tree Classifier e Random Forest Classifier.

Estrutura do Repositório
data/: Pasta contendo os conjuntos de dados utilizados.
notebooks/: Notebooks Jupyter com exemplos de código.
scripts/: Scripts Python para treinamento e avaliação dos modelos.

Uso
Execute os scripts train_model_decision_tree.py e train_model_random_forest.py para treinar os modelos.
Os modelos treinados são avaliados quanto à acurácia e as métricas são exibidas no console.
Exemplo de Código
python
Copiar código
# Exemplo de código Python utilizando Spark para treinamento e avaliação de modelos
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# Divisão dos dados em conjunto de treino e teste
train, test = df.randomSplit([0.7, 0.3])

# Treinamento do modelo Decision Tree
dt = DecisionTreeClassifier(labelCol="labelIndex", featuresCol="features")
model_dt = dt.fit(train)

# Avaliação do modelo Decision Tree
predictions_dt = model_dt.transform(test)
evaluator_dt = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
accuracy_dt = evaluator_dt.evaluate(predictions_dt)

print("Acurácia do Decision Tree: ", accuracy_dt)

# Treinamento do modelo Random Forest
rf = RandomForestClassifier(labelCol="labelIndex", featuresCol="features")
model_rf = rf.fit(train)

# Avaliação do modelo Random Forest
predictions_rf = model_rf.transform(test)
evaluator_rf = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
accuracy_rf = evaluator_rf.evaluate(predictions_rf)

print("Acurácia do Random Forest: ", accuracy_rf)
