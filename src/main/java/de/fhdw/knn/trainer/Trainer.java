package de.fhdw.knn.trainer;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.io.Exporter;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.Adjustments;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.visualization.HeatmapData;
import de.fhdw.knn.visualization.HeatmapWindow;
import de.fhdw.knn.visualization.SankeyLiveView;
import javafx.application.Platform;

import java.util.stream.IntStream;

public class Trainer {

    public final Network network;
    public final int maxEpochs;
    public final boolean shuffleEpoch;
    public int batchSize;

    public final LossFunction lossFunction;
    public final StopFunction stopFunction;
    public final OptimizationFunction optimizationFunction;

    // Einstellung: Alle wie viele Epochen soll die Heatmap/Sankey aktualisiert werden?
    public static int HEATMAP_INTERVAL = 5;
    public static int SANKEY_INTERVAL = 10;

    public Trainer(Network network, int maxEpochs, boolean shuffleEpoch, int batchSize, LossFunction lossFunction, StopFunction stopFunction, OptimizationFunction optimizationFunction) {
        this.network = network;
        this.maxEpochs = maxEpochs;
        this.shuffleEpoch = shuffleEpoch;
        this.batchSize = batchSize;
        this.lossFunction = lossFunction;
        this.stopFunction = stopFunction;
        this.optimizationFunction = optimizationFunction;
    }

    public void train(DataSet data) {
        train(data, null, 0);
    }

    public void train(DataSet data, String export, int mod) {

        HeatmapWindow heatmapWindow = null;
        SankeyLiveView sankeyView = null;

        if (HEATMAP_INTERVAL > 0) {
            heatmapWindow = new HeatmapWindow();
        }

        if (SANKEY_INTERVAL > 0) {
            try {
                Platform.startup(() -> {});
            } catch (IllegalStateException ignored) {} // Falls es schon läuft

            sankeyView = new SankeyLiveView();
            sankeyView.show(this.network);
        }

        double totalLoss = Double.NaN;

        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            int humanEpoch = epoch + 1;

            System.out.println("Epoch: " + humanEpoch);
            optimizationFunction.epoch(epoch, totalLoss);
            totalLoss = trainEpoch(data);

            // Heatmap Update
            if (HEATMAP_INTERVAL > 0 && heatmapWindow != null) {
                if (humanEpoch == 1 || humanEpoch % HEATMAP_INTERVAL == 0 || humanEpoch == maxEpochs) {
                    heatmapWindow.addEpoch("Epoche " + humanEpoch, new HeatmapData(this.network));
                    System.out.println("Heatmap-Update bei Epoche " + humanEpoch);
                }
            }

            // Sankey Update
            if (SANKEY_INTERVAL > 0 && sankeyView != null) {
                if (humanEpoch == 1 || humanEpoch % SANKEY_INTERVAL == 0 || humanEpoch == maxEpochs) {
                    sankeyView.update(this.network, humanEpoch);
                    System.out.println("Sankey-Update bei Epoche " + humanEpoch);
                }
            }

            // Export
            if (export != null && mod > 0 && (epoch % mod == 0 || epoch == maxEpochs - 1)) {
                Exporter.export(network, export.formatted(humanEpoch));
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
                    adjustments[offset] = optimizationFunction.compute(data.inputs[index], data.outputs[index], batchSize);
                });
                for (Adjustments adjustment : adjustments) {
                    adjustment.adjust(network);
                }
            }
        } else {
            for (int i = 0; i < data.size; i += 1) {
                optimizationFunction.compute(data.inputs[i], data.outputs[i], 1).adjust(network);
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
