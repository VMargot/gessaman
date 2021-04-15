from gessaman.gessaman import Gessaman
from sklearn.datasets import load_boston, load_diabetes
from sklearn.metrics import r2_score


if __name__ == '__main__':
    X, y = load_boston(return_X_y=True)
    g = Gessaman()
    g.fit(X, y)
    pred = g.predict(X)
    print('Boston: ', r2_score(y, pred))
    # Boston:  0.37748412469794135

    X, y = load_diabetes(return_X_y=True)
    g = Gessaman()
    g.fit(X, y)
    pred = g.predict(X)
    print('Diabetes: ', r2_score(y, pred))
    # Diabetes:  0.444621563690068
