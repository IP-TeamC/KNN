package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.Network;

/**
 * Die Optimierungsfunktion ermittelt die für das Training notwendigen Anpassungen des Netzwerks.<br>
 * Die Methode {@link OptimizationFunction#epoch(int, double)} wird vor jeder Epoche aufgerufen,
 * um notwendige Anpassungen wie das Setzen der Learning Rate für die Epoche durchzuführen.
 * {@link OptimizationFunction#compute(Network, double[], double[], int)} berechnet die Anpassungen des Netzwerks für einen Datensatz.
 */
public interface OptimizationFunction {

    /**
     * Wird vor jeder Epoche aufgerufen, um z.B. die Learning Rate anzupassen.
     *
     * @param epoch        Aktuelle Epoche (beginnt mit 1)
     * @param previousLoss Total-Loss der vorherigen Epoche (bei 1. Epoche NaN)
     */
    void epoch(int epoch, double previousLoss);

    /**
     * Berechnet die Anpassungen des Netzwerks für eine Zeile des Datensatzes
     *
     * @param network   Netzwerk, das optimiert werden soll
     * @param input     Eingabe-Zeile aus dem Datensatz
     * @param output    Ausgabe-Zeile aus dem Datensatz
     * @param batchSize Batch-Size für z.B. Mini-Batching (Learning Rate wird durch batchSize geteilt)
     * @return berechnete Anpassungen des Netzwerks für die übergebene Zeile des Datensatzes
     */
    Adjustments compute(Network network, double[] input, double[] output, int batchSize);

}
