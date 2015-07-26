%============================================
% Ejemplo Clasificador Naive Bayes
% Autor: Alexander Gómez Villa
% Se usa la base de datos iris Setosa
%============================================


%% carga los datos

load fisheriris % Carga base de datos Iris Setosa, por defecto de Matlab


%% Entrenamiento
Modelo = fitNaiveBayes(meas,species);% Al no especificar Kernel se usa Gaussiano por defecto


%% Evaluacion
prediccion = Modelo.predict(meas); % Prediccion usando conjunto de entrenamiento

%% Resultados
cMat1 = confusionmat(species,prediccion) % Imprime matriz de confusion