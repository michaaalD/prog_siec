Perceptron to prosty model neuronowy używany do klasyfikacji binarnej.
Inicjalizujemy go podając współczynnik uczenia (eta) oraz liczbę epok (epochs).
Metoda train trenuje perceptron na danych treningowych X z oczekiwanymi etykietami y.
Wagi (self.w_) są aktualizowane w każdej epoce na podstawie błędów klasyfikacji.
Metoda net_input oblicza wartość wejścia do perceptronu.
Metoda predict dokonuje klasyfikacji na podstawie wartości wejścia.
Jeśli masz jeszcze jakieś pytania, śmiało pytaj! 😊


# implementacja Perceptronu
import numpy as np



class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
