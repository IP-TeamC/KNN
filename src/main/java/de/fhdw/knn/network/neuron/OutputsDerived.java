package de.fhdw.knn.network.neuron;

public record OutputsDerived(double[][] output, double[][] derived) {

    public double[] lastOutput() {
        return output[output.length - 1];
    }

}
