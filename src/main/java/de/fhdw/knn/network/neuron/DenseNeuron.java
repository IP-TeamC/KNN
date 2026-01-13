package de.fhdw.knn.network.neuron;

import de.fhdw.knn.data.Pair;
import de.fhdw.knn.network.activation.ActivationFunction;

public class DenseNeuron extends AbstractDenseNeuron {

	public double bias = 0.000_000_000_1;
	public ActivationFunction activationFunction;

	@Override
	public Pair<Double, Double> compute(double[] input) {
		double weightedSum = bias;
		for (int i = 0; i < input.length; i++) {
			if (incoming[i].guard) {
				weightedSum += incoming[i].weight * input[i];
			}
		}

		double output = activationFunction.calc(weightedSum);
		double derived = activationFunction.derived(weightedSum, output);
		return new Pair<>(output, derived);
	}

}
