from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

si = SimpleImputer(
    missing_values=np.nan,  # los valores que faltan son del tipo ``np.nan`` (Pandas estándar)
    strategy= 'mean',
    #fill_value= 0,# la estrategia elegida es cambiar el valor faltante por una constante
      # la constante que se usará para completar los valores faltantes es un int64 = 0
    verbose=0,
    copy=True)


# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')


class OutlierExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('init')

    def fit(self, X, Y):
        return self

    def transform(self, X, Y):
        # Primero copiamos el dataframe de datos de entrada 'X'
        X2 = si.fit_transform(X)
        Y2 = Y.to_numpy()
        iso = IsolationForest(n_estimators=400,contamination=0.3,random_state=42)
        yhat = iso.fit_predict(X2)
        mask = yhat != -1
        X2, Y2 = X2[mask, :], Y2[mask]
        print(X2.shape)
        print(Y2.shape)
        X_train2 = pd.DataFrame.from_records(data=X2,columns=X.columns)
        Y_train2 = Y2

        return (X_train2, Y_train2)
