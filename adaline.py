Definicja klasy: AdalineGD:
Ta klasa reprezentuje model Adaline z nauką gradientową.
Konstruktor (__init__) inicjalizuje model parametrami:
eta: Współczynnik uczenia (domyślnie 0,01).
epochs: Liczba iteracji treningowych (domyślnie 50).
Metoda train trenuje model Adaline za pomocą nauki gradientowej:
Inicjalizuje wagi (self.w_) na zera.
Dla każdej epoki oblicza wyjście, aktualizuje wagi i oblicza koszt (błąd średniokwadratowy).
Wagi są aktualizowane za pomocą iloczynu skalarnego cech wejściowych (X) i błędów (y - output).
Koszt dla każdej epoki jest przechowywany w self.cost_.
Metoda net_input oblicza wejście netto (ważoną sumę cech).
Metoda activation zwraca wejście netto.
Metoda predict klasyfikuje próbki na podstawie funkcji aktywacji:
Jeśli aktywacja jest większa lub równa 0, przewiduje klasę 1; w przeciwnym razie przewiduje klasę -1.
Użycie w Twoim kodzie:
Zastosowałeś model Adaline do dwóch różnych zadań klasyfikacji binarnej:
Setosa vs. Versicolor (używając cech długości działki kielicha i długości płatka).
Virginica vs. Versicolor (używając cech szerokości działki kielicha i szerokości płatka).
Dodatkowo połączyłeś Setosę i Virginicę (używając cech szerokości działki kielicha i szerokości płatka).
Uwaga:
Komentarz opisz funkcje klasy sugeruje, że zamierzasz opisać metody klasy bardziej szczegółowo. Możesz dodać odpowiednie opisy dla każdej metody, aby zwiększyć czytelność kodu.


import numpy as np

class AdalineGD(object):
    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)
