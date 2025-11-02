package de.fhdw.knn.network.neuron;

import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.activation.SigmoidActivationFunction;
import lombok.ToString;

@ToString
public class DenseNeuron extends Neuron {

    @ToString.Exclude
    public double outputCache;
    @ToString.Exclude
    public double outputDerivedCache;

    public double bias = 0;
    public Connection[] incoming;
    @ToString.Exclude
    public ActivationFunction activationFunction = SigmoidActivationFunction.DEFAULT;

    @Override
    public void compute() {
        double weightedSum = bias;
        for (Connection connection : incoming) {
            if (connection.guard) {
                weightedSum += connection.weight * connection.inputNeuron.output();
            }
        }
        outputCache = activationFunction.calc(weightedSum);
        outputDerivedCache = activationFunction.derived(weightedSum);
    }

    @Override
    public double output() {
        return outputCache;
    }

}
