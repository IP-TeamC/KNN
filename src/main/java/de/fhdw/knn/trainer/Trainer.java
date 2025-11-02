package de.fhdw.knn.trainer;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.StopFunction;

public class Trainer {

    public final int maxEpochs;
    public final Network network;

    public final LossFunction lossFunction;
    public final StopFunction stopFunction;
    public final OptimizationFunction optimizationFunction;

    public Trainer(Network network, int maxEpochs, LossFunction lossFunction, StopFunction stopFunction, OptimizationFunction optimizationFunction) {
        this.network = network;
        this.maxEpochs = maxEpochs;
        this.lossFunction = lossFunction;
        this.stopFunction = stopFunction;
        this.optimizationFunction = optimizationFunction;
    }

    public void train(DataSet data) {
        optimizationFunction.init();
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            System.out.println("Epoch: " + epoch);
            double totalLoss = trainEpoch(data);
            if (stopFunction.isFinished(totalLoss)) {
                break;
            }
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
