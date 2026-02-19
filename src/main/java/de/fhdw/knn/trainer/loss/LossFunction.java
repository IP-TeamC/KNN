package de.fhdw.knn.trainer.loss;

/**
 * Die Verlustfunktion ist ein Maß für den Fehler des Modells/KNNs im Vergleich zum Datensatz.<br>
 * Für den Optimierungsalgorithmus wird lediglich {@link LossFunction#derivedLoss(double[], double[], int)} benötigt.
 * Diese Methode leitet die Verlustfunktion für ein Output-Neuron nach predicted (der Vorhersage/Output) ab.<br>
 * Die Methoden {@link LossFunction#loss(double[], double[])} sowie {@link LossFunction#totalLoss(double[][], double[][])}
 * ermitteln den Loss für eine Eingabe-Zeile bzw. den gesamten Datensatz.
 */
public interface LossFunction {

    LossFunction MEAN_SQUARED_ERROR = new MeanSquaredError();
    LossFunction CROSS_ENTROPY_LOSS = new BinaryCrossEntropyLoss();
    LossFunction MEAN_ABSOLUTE_ERROR = new MeanAbsoluteError();

    double loss(double[] expected, double[] predicted);

    double derivedLoss(double[] expected, double[] predicted, int neuron);

    default double totalLoss(double[][] expected, double[][] predicted) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum += loss(expected[i], predicted[i]);
        }
        return sum / expected.length;
    }

}
