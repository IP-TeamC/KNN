package de.fhdw.knn.trainer;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.io.Exporter;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.StopFunction;

public class Trainer {

    public final Network network;
    public final int maxEpochs;
    public final boolean shuffleEpoch;

    public final LossFunction lossFunction;
    public final StopFunction stopFunction;
    public final OptimizationFunction optimizationFunction;

    public Trainer(Network network, int maxEpochs, boolean shuffleEpoch, LossFunction lossFunction, StopFunction stopFunction, OptimizationFunction optimizationFunction) {
        this.network = network;
        this.maxEpochs = maxEpochs;
        this.shuffleEpoch = shuffleEpoch;
        this.lossFunction = lossFunction;
        this.stopFunction = stopFunction;
        this.optimizationFunction = optimizationFunction;
    }

    public void train(DataSet data) {
        train(data, null, 0);
    }

    public void train(DataSet data, String export, int mod) {
        optimizationFunction.init();
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            System.out.println("Epoch: " + epoch);
            double totalLoss = trainEpoch(data);

            if (export != null && (epoch % mod == 0 || epoch == maxEpochs - 1))
                Exporter.export(network, export.formatted(epoch));

            if (stopFunction.isFinished(totalLoss)) break;
            else if (shuffleEpoch) data.shuffle(epoch);
        }
    }

    private double trainEpoch(DataSet data) {
        for (int i = 0; i < data.size; i++) {
            optimizationFunction.compute(data.inputs[i], data.outputs[i]).adjust(network);
        }

        double[][] predicted = network.predict(data.inputs);
        double totalLoss = lossFunction.totalLoss(data.outputs, predicted);
        System.out.println("Loss: " + totalLoss);
        return totalLoss;
    }

}
