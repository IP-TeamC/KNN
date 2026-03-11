package de.fhdw.knn.network.activation;

import java.util.List;

/**
 * Interface für die Implementierung der Aktivierungsfunktionen von Neuronen.
 * Eine Aktivierungsfunktion berechnet für eine Eingabe (gewichtete Summe der Verbindungen ins Neuron) eine Ausgabe (Aktivierung).
 * Diese Ausgabe wird mit {@link ActivationFunction#calc(double)} berechnet.<br>
 * Für den Optimierungsalgorithmus Gradient Descent muss zusätzlich die Ableitung der Aktivierungsfunktion an einem bestimmten Punkt berechnet werden.
 * Die Ableitung wird mit {@link ActivationFunction#derived(double, double)} berechnet, wobei der zweite Parameter calc die bereits berechnete Aktivierung mitliefert (zur Optimierung der Performance in einigen Fällen).
 */
public interface ActivationFunction {

    /**
     * a(x) = 1 / (1 + e^(-x))
     *
     * @see ActivationFunction
     */
    ActivationFunction SIGMOID = new SigmoidActivationFunction();

    /**
     * a(x) = max(x, 0)<br>
     * Vergleichbar mit {@link ActivationFunction#SWISH}
     *
     * @see ActivationFunction
     */
    ActivationFunction RELU = new ReLUActivationFunction();

    /**
     * a(x) = x
     *
     * @see ActivationFunction
     */
    ActivationFunction LINEAR = new LinearActivationFunction();

    /**
     * a(x) = tanh(x)
     *
     * @see ActivationFunction
     */
    ActivationFunction TANH = new TanhActivationFunction();

    /**
     * a(x) = x / (1 + e^(-x))<br>
     * Vergleichbar mit {@link ActivationFunction#RELU}
     *
     * @see ActivationFunction
     */
    ActivationFunction SWISH = new SwishActivationFunction();

    /**
     * a(x) = ln(1 + e^x)
     */
    ActivationFunction SOFTPLUS = new SoftplusActivationFunction();

    /**
     * a(x) = (sin(x))^2 + x
     */
    ActivationFunction SNAKE = new SnakeActivationFunction();

    /**
     * a(x) = sin(x)
     */
    ActivationFunction SIN = new SinActivationFunction();

    /**
     * Liste aller Aktivierungsfunktionen,
     * mit welcher über den Index eine Zuordnung der Aktivierungsfunktionen für den Import/Export von Netzwerken ermöglicht wird.
     * Eine Änderung der Reihenfolge führt zur Inkompatibilität der Datenformate beim Import/Export.
     */
    List<ActivationFunction> FUNCTIONS = List.of(SIGMOID, RELU, LINEAR, TANH, SWISH, SOFTPLUS, SNAKE, SIN);

    /**
     * Berechnet die Aktivierung eines Neurons
     *
     * @param input gewichtete Summe der Verbindungen ins Neuron
     * @return Aktivierung/Ausgabe des Neurons
     * @see ActivationFunction
     */
    double calc(double input);

    /**
     * Berechnet Ableitung der Aktivierung eines Neurons
     *
     * @param input gewichtete Summe der Verbindungen ins Neuron
     * @param calc  bereits berechnete Aktivierung des Neurons ({@link ActivationFunction#calc(double)})
     * @return Ableitung der Aktivierungsfunktion des Neurons für die gegebene Eingabe
     * @see ActivationFunction
     */
    double derived(double input, double calc);

}