package de.fhdw.knn.network.neuron;

import de.fhdw.knn.network.activation.ActivationFunction;

public class DenseNeuron extends Neuron {

    public double outputCache;
    public double outputDerivedCache;

    public double bias = 0;
    public Connection[] incoming;
    public ActivationFunction activationFunction = ActivationFunction.SIGMOID;

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

    @Override
    public double output() {
        return outputCache;
    }

}
