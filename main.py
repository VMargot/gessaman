from gessaman.gessaman import Gessaman
from sklearn.datasets import load_boston, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    g = Gessaman()
    g.fit(X_train, y_train)
    pred = g.predict(X_test)
    print('Boston % of bad points: ', sum(1 - np.isfinite(pred)) / len(pred) * 100)
    pred = np.nan_to_num(pred)
    print('Boston: ', r2_score(y_test, pred))
    # Boston:  0.37748412469794135

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)
    g = Gessaman()
    g.fit(X_train, y_train)
    pred = g.predict(X_test)
    print('Diabetes % of bad points: ', sum(1 - np.isfinite(pred)) / len(pred))
    pred = np.nan_to_num(pred)
    print('Diabetes: ', r2_score(y_test, pred))
    # Diabetes:  0.444621563690068
