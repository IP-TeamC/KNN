package de.fhdw.knn.trainer;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.io.Exporter;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.Adjustments;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.visualization.ViewManager;
import lombok.Setter;

import java.util.stream.IntStream;

/**
 * Zentrale Steuerungsklasse für das Training des Netzwerks.<br>
 * Führt die Epochen des Trainings aus ({@link OptimizationFunction}),
 * steuert die vorzeitige Beendigung ({@link StopFunction}),
 * gibt den Loss aus ({@link LossFunction}),
 * visualisiert das Netzwerk ({@link ViewManager}) und
 * umfasst alle Parameter des Trainings.
 */
public class Trainer {

    private final Network network;
    private final int maxEpochs;
    private final boolean shuffleEpoch;
    private final int batchSize;

    private final LossFunction lossFunction;
    private final StopFunction stopFunction;
    private final OptimizationFunction optimizationFunction;

    @Setter
    private ViewManager viewManager;

    /**
     * Erzeugt einen neuen Trainer ohne Visualisierung
     *
     * @param network              das zu trainierende Netzwerk
     * @param maxEpochs            die maximale Anzahl Epochen, für die trainiert werden soll
     * @param shuffleEpoch         gibt an, ob die Reihenfolge der Zeilen im Datensatz nach jeder Epoche zufällig durchmischt werden soll (empfohlen)
     * @param batchSize            Anzahl gleichzeitig zu verarbeitender Zeilen des Datensatzes ohne Anpassung nach jeder Zeile (Mini-Batching)
     * @param lossFunction         Verlustfunktion für die Ausgabe des Loss (nicht zum Training selbst, diese muss in der OptimizationFunction definiert werden)
     * @param stopFunction         Vorzeitige Beendigung des Trainings
     * @param optimizationFunction Optimierungsfunktion (in der Regel {@link GradientDescent} )
     * @see Trainer
     */
    public Trainer(Network network, int maxEpochs, boolean shuffleEpoch, int batchSize, LossFunction lossFunction, StopFunction stopFunction, OptimizationFunction optimizationFunction) {
        this(network, maxEpochs, shuffleEpoch, batchSize, lossFunction, stopFunction, optimizationFunction, null);
    }

    /**
     * Erzeugt einen neuen Trainer
     *
     * @param network              das zu trainierende Netzwerk
     * @param maxEpochs            die maximale Anzahl Epochen, für die trainiert werden soll
     * @param shuffleEpoch         gibt an, ob die Reihenfolge der Zeilen im Datensatz nach jeder Epoche zufällig durchmischt werden soll (empfohlen)
     * @param batchSize            Anzahl gleichzeitig zu verarbeitender Zeilen des Datensatzes ohne Anpassung nach jeder Zeile (Mini-Batching)
     * @param lossFunction         Verlustfunktion für die Ausgabe des Loss (nicht zum Training selbst, diese muss in der OptimizationFunction definiert werden)
     * @param stopFunction         Vorzeitige Beendigung des Trainings
     * @param optimizationFunction Optimierungsfunktion (in der Regel {@link GradientDescent} )
     * @param viewManager      Visualisierung des Netzwerks während des Trainings
     * @see Trainer
     */
    public Trainer(Network network, int maxEpochs, boolean shuffleEpoch, int batchSize, LossFunction lossFunction, StopFunction stopFunction, OptimizationFunction optimizationFunction, ViewManager viewManager) {
        this.network = network;
        this.maxEpochs = maxEpochs;
        this.shuffleEpoch = shuffleEpoch;
        this.batchSize = batchSize;
        this.lossFunction = lossFunction;
        this.stopFunction = stopFunction;
        this.optimizationFunction = optimizationFunction;
        this.viewManager = viewManager;
    }

    /**
     * Führt für den gegebenen Datensatz das gesamte Training entsprechend der eigenen Parameter durch
     */
    public void train(DataSet data) {
        train(data, null, 0);
    }

    /**
     * Führt für den gegebenen Datensatz das gesamte Training entsprechend der eigenen Parameter durch.<br>
     * Export des KNN in regelmäßigen Abstände.
     *
     * @param export Dateipfad des exportierten KNNs (mit %d als Platzhalter für die aktuelle Epoche)
     * @param mod    Abstand zwischen Exporten (exportiert immer dann, wenn die aktuelle Epoche durch mod teilbar ist)
     */
    public void train(DataSet data, String export, int mod) {
        double totalLoss = Double.NaN;

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            System.out.println("Epoch: " + epoch);
            optimizationFunction.epoch(epoch, totalLoss);
            totalLoss = trainEpoch(data);

            // Visualization
            if (viewManager != null) {
                viewManager.nextEpoch(epoch, network, maxEpochs);
            }

            // Export
            if (export != null && mod > 0 && (epoch % mod == 0 || epoch == maxEpochs - 1)) {
                Exporter.export(network, export.formatted(epoch));
            }

            if (stopFunction.isFinished(totalLoss)) break;
            if (shuffleEpoch) data.shuffle(epoch);
        }
    }

    private double trainEpoch(DataSet data) {
        if (batchSize > 1) {
            Adjustments[] adjustments = new Adjustments[batchSize];
            for (int i = 0; i < data.size; i += batchSize) {
                int base = i;
                int limit = Math.min(batchSize, data.size - i);
                IntStream.range(0, limit).parallel().forEach(offset -> {
                    int index = base + offset;
                    adjustments[offset] = optimizationFunction.compute(network, data.inputs[index], data.outputs[index], batchSize);
                });
                for (Adjustments adjustment : adjustments) {
                    adjustment.adjust(network);
                }
            }
        } else {
            for (int i = 0; i < data.size; i += 1) {
                optimizationFunction.compute(network, data.inputs[i], data.outputs[i], 1).adjust(network);
            }
        }

        if (lossFunction != null) {
            double[][] predicted = network.predict(data.inputs);
            double totalLoss = lossFunction.totalLoss(data.outputs, predicted);
            System.out.println("Loss: " + totalLoss);
            return totalLoss;
        } else {
            return Double.NaN;
        }
    }
}
