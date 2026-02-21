package de.fhdw.knn.trainer.loss;

/**
 * Die Verlustfunktion ist ein Maß für den Fehler des Modells/KNNs im Vergleich zum Datensatz.<br>
 * Für den Optimierungsalgorithmus wird lediglich {@link LossFunction#derivedLoss(double[], double[], int)} benötigt.
 * Diese Methode leitet die Verlustfunktion für ein Output-Neuron nach predicted (der Vorhersage/Output) ab.<br>
 * Die Methoden {@link LossFunction#loss(double[], double[])} sowie {@link LossFunction#totalLoss(double[][], double[][])}
 * ermitteln den Loss für eine Eingabe-Zeile bzw. den gesamten Datensatz.
 */
public interface LossFunction {

    /**
     * Mittlere quadratische Abweichung
     */
    LossFunction MEAN_SQUARED_ERROR = new MeanSquaredError();
    /**
     * Binary Cross-Entropy Loss
     */
    LossFunction CROSS_ENTROPY_LOSS = new BinaryCrossEntropyLoss();
    /**
     * Mittlere absolute Abweichung
     */
    LossFunction MEAN_ABSOLUTE_ERROR = new MeanAbsoluteError();

    /**
     * Berechnet den mittleren Verlust über alle Output-Neuronen für einen Datensatz
     */
    double loss(double[] expected, double[] predicted);

    /**
     * Berechnet die Ableitung der Verlust-Funktion für ein Output-Neuron für einen Datensatz
     */
    double derivedLoss(double[] expected, double[] predicted, int neuron);

    /**
     * Berechnet den mittleren Verlust über alle Output-Neuronen und mehrere Datensätze.<br>
     * Diese Methode muss in der Regel für eine Implementierung dieses Interfaces nicht implementiert werden,
     * sondern wird automatisch aus {@link LossFunction#loss(double[], double[])} abgeleitet.
     */
    default double totalLoss(double[][] expected, double[][] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum += loss(expected[i], predicted[i]);
        }
        return sum / expected.length;
    }

}
