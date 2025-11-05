package de.fhdw.knn.network.neuron;

import de.fhdw.knn.network.activation.ActivationFunction;

public class DenseNeuron extends Neuron {

    @Deprecated
    public double outputCache;
    @Deprecated
    public double outputDerivedCache;

    public double bias = 0;
    public Connection[] incoming;
    public ActivationFunction activationFunction = ActivationFunction.SIGMOID;

    @Deprecated
    @Override
    public void compute() {
        double weightedSum = bias;
        for (Connection connection : incoming) {
            if (connection.guard) {
                weightedSum += connection.weight * connection.inputNeuron.output();
            }
        }
        outputCache = activationFunction.calc(weightedSum);
        outputDerivedCache = activationFunction.derived(weightedSum, outputCache);
    }

    public double computeWeightedSum(double[] input) {
        double weightedSum = 0;
        for (int i = 0; i < input.length; i++) {
            weightedSum += incoming[i].weight * input[i];
        }
        return weightedSum;
    }

    @Override
    public double output() {
        return outputCache;
    }

}
