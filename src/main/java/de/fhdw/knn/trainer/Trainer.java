package de.fhdw.knn.trainer;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.io.Exporter;
import de.fhdw.knn.network.neuron.OutputsDerived;
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

    /**
     * Zu trainierendes Netzwerk
     */
    private final Network network;
    /**
     * Maximale Anzahl zu trainierender Epochen
     */
    private final int maxEpochs;
    /**
     * Gibt an, ob die Reihenfolge der Zeilen im Datensatz vor jeder Epoche zufällig durchmischt werden soll
     */
    private final boolean shuffleEpoch;
    /**
     * Größe der zu berechnenden Gewichts-/Bias-Anpassungen vor einer tatsächlichen Anpassung (z.B. für Mini-Batching, wenn &gt;1)
     */
    private final int batchSize;

    /**
     * Zu verwendende Verlustfunktion (nur zur Ausgabe des aktuellen Verlusts/Loss und für die StopFunction)<br>
     * Kann null sein, um die Ausgabe zu deaktivieren
     */
    private final LossFunction lossFunction;
    /**
     * Zu verwendende Funktion für das vorzeitige Beenden des Trainings
     */
    private final StopFunction stopFunction;
    /**
     * Zu verwendende Optimierungsfunktion (nur Gradient Descent ist bisher verfügbar)
     */
    private final OptimizationFunction optimizationFunction;
    /**
     * Buffer für die Adjustments, da diese viele große Arrays enthalten und nicht für jede
     * Ausführung des Optimierungsalgorithmus neu allokiert und vom GC wieder gelöscht werden müssen
     */
    private final Adjustments[] adjustmentsBuffer;
    /**
     * Buffers für die Neuron-Aktivierungen/Ausgaben, da diese aus vielen großen Arrays bestehen und nicht für jede
     * Ausführung des Optimierungsalgorithmus neu allokiert und vom GC wieder gelöscht werden müssen
     */
    private final OutputsDerived[] buffers;

    /**
     * Verwaltung der Visualisierung während des Trainings
     */
    @Setter
    private ViewManager viewManager;

    /**
     * Erzeugt einen neuen Trainer ohne Visualisierung
     *
     * @param network              das zu trainierende Netzwerk
     * @param maxEpochs            die maximale Anzahl Epochen, für die trainiert werden soll
     * @param shuffleEpoch         gibt an, ob die Reihenfolge der Zeilen im Datensatz nach jeder Epoche zufällig durchmischt werden soll (empfohlen)
     * @param batchSize            Anzahl gleichzeitig zu verarbeitender Zeilen des Datensatzes ohne Anpassung nach jeder Zeile (Mini-Batching)
     * @param lossFunction         Verlustfunktion für die Ausgabe des Loss und für die StopFunction (nicht zum Training selbst, diese muss in der OptimizationFunction definiert werden)
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
     * @param lossFunction         Verlustfunktion für die Ausgabe des Loss und für die StopFunction (nicht zum Training selbst, diese muss in der OptimizationFunction definiert werden)
     * @param stopFunction         Vorzeitige Beendigung des Trainings
     * @param optimizationFunction Optimierungsfunktion (in der Regel {@link GradientDescent} )
     * @param viewManager          Visualisierung des Netzwerks während des Trainings
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

        this.adjustmentsBuffer = new Adjustments[batchSize];
        this.buffers = new OutputsDerived[batchSize];
        for (int i = 0; i < batchSize; i++) {
            adjustmentsBuffer[i] = Adjustments.generateEmpty(network);
            buffers[i] = OutputsDerived.generateEmpty(network);
        }
    }

    /**
     * Führt für den gegebenen Datensatz das gesamte Training entsprechend der eigenen Parameter und inkl. Anpassung des Netzwerks durch
     *
     * @param data Gesamter Trainings-Datensatz
     * @see Trainer#train(DataSet, String, int)
     */
    public void train(DataSet data) {
        train(data, null, 0);
    }

    /**
     * Führt für den gegebenen Datensatz das gesamte Training entsprechend der eigenen Parameter durch.<br>
     * Das tatsächliche Training einer Epoche findet in {@link Trainer#trainEpoch(DataSet)} statt.<br>
     * Bei export != null findet ein Export des KNN in regelmäßigen Abständen statt.
     *
     * @param data   Gesamter Trainings-Datensatz
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
            if (export != null && mod > 0 && (epoch % mod == 0 || epoch == maxEpochs)) {
                Exporter.export(network, export.formatted(epoch));
            }

            if (stopFunction.isFinished(totalLoss)) break;
            if (shuffleEpoch) data.shuffle(epoch);
        }
    }

    /**
     * Training des Datensatzes für eine Epoche.
     * Berechnet für jede Zeile des Datensatzes die Anpassungen des Netzwerks und wendet diese an (unter Beachtung der batchSize).
     *
     * @param data Trainings-Datensatz
     * @return Mit der LossFunction ermittelter Verlust für den gesamten Datensatz
     */
    private double trainEpoch(DataSet data) {
        if (batchSize > 1) {
            for (int i = 0; i < data.size; i += batchSize) {
                int base = i;
                int limit = Math.min(batchSize, data.size - i);
                IntStream.range(0, limit).parallel().forEach(offset -> {
                    int index = base + offset;
                    optimizationFunction.compute(adjustmentsBuffer[offset], buffers[offset],
                            network, data.inputs[index], data.outputs[index], batchSize);
                });
                IntStream.range(0, network.denseLayers.length).parallel().forEach(layer -> {
                    for (int offset = 0; offset < limit; offset++) {
                        adjustmentsBuffer[offset].adjust(network, layer);
                    }
                });
            }
        } else {
            Adjustments adjustments = adjustmentsBuffer[0];
            OutputsDerived buffer = buffers[0];
            for (int i = 0; i < data.size; i += 1) {
                optimizationFunction.compute(adjustments, buffer,
                        network, data.inputs[i], data.outputs[i], 1);
                adjustments.adjust(network);
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
