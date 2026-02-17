package de.fhdw.knn.trainer;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.io.Exporter;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.Adjustments;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.visualization.LiveViewManager;
import lombok.Setter;

import java.util.stream.IntStream;

public class Trainer {

    public final Network network;
    public final int maxEpochs;
    public final boolean shuffleEpoch;
    public int batchSize;

    public final LossFunction lossFunction;
    public final StopFunction stopFunction;
    public final OptimizationFunction optimizationFunction;

    @Setter
    private LiveViewManager liveViewManager;

    public Trainer(Network network, int maxEpochs, boolean shuffleEpoch, int batchSize, LossFunction lossFunction, StopFunction stopFunction, OptimizationFunction optimizationFunction) {
        this(network, maxEpochs, shuffleEpoch, batchSize, lossFunction, stopFunction, optimizationFunction, null);
    }

    public Trainer(Network network, int maxEpochs, boolean shuffleEpoch, int batchSize, LossFunction lossFunction, StopFunction stopFunction, OptimizationFunction optimizationFunction, LiveViewManager liveViewManager) {
        this.network = network;
        this.maxEpochs = maxEpochs;
        this.shuffleEpoch = shuffleEpoch;
        this.batchSize = batchSize;
        this.lossFunction = lossFunction;
        this.stopFunction = stopFunction;
        this.optimizationFunction = optimizationFunction;
        this.liveViewManager = liveViewManager;
    }

    public void train(DataSet data) {
        train(data, null, 0);
    }

    public void train(DataSet data, String export, int mod) {
        double totalLoss = Double.NaN;

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            System.out.println("Epoch: " + epoch);
            optimizationFunction.epoch(epoch, totalLoss);
            totalLoss = trainEpoch(data);

            // Visualization
            if (liveViewManager != null) {
                liveViewManager.nextEpoch(epoch, network, maxEpochs);
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
