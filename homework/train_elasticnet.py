#
# Busque los mejores parametros de un modelo ElasticNet para predecir 
# la calidad del vino usando el dataset de calidad del vino tinto de UCI 
#
# Considere los siguientes valores de los hiperparámetros y obtenga el  
# mejor modelo
# (alpha, l1_ratio) 
#       (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

# Importe las librerías 
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.print_metrics import print_metrics
from homework.src._internals.save_model_if_better import save_model_if_better

x_train, x_test, y_train, y_test = prepare_data(
    file_path="winequality-red.csv",
    test_size=0.25,
    random_state=123456)

# Entrenar modelo 
estimator = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=123456)

estimator.fit(x_train, y_train)

mse, mae , r2 = calculate_metrics (estimator, x_train, y_train)
print_metrics("Training metrics", mse, mae, r2)

mse, mae , r2 = calculate_metrics (estimator, x_test, y_test)
print_metrics("Testing metrics", mse, mae, r2) 

save_model_if_better(estimator, x_test, y_test)