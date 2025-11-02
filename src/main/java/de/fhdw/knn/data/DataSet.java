package de.fhdw.knn.data;

import java.util.Random;

public class DataSet {

    public final int size;
    public final int inputSize;
    public final int outputSize;

    public double[][] inputs;
    public double[][] outputs;

    private Normalizer normalizerInputs;
    private Normalizer normalizerOutputs;

    public DataSet(double[][] inputs, double[][] outputs) {
        if (inputs.length != outputs.length) {
            throw new IllegalArgumentException("input.length != output.length");
        }

        this.size = inputs.length;
        this.inputSize = inputs[0].length;
        this.outputSize = outputs[0].length;

        this.inputs = inputs;
        this.outputs = outputs;
    }

    public void shuffle(long seed) {
        Random random = new Random(seed);
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] output = outputs[i];
            int pos = random.nextInt(inputs.length);

            inputs[i] = inputs[pos];
            outputs[i] = outputs[pos];

            inputs[pos] = input;
            outputs[pos] = output;
        }
    }

    public TrainTestSplit shuffleAndSplit(long seed, double testShare) {
        shuffle(seed);
        int testSize = (int) (testShare * size);
        int trainSize = size - testSize;

        double[][] inputsTrain = new double[trainSize][];
        double[][] outputsTrain = new double[trainSize][];
        System.arraycopy(inputs, 0, inputsTrain, 0, trainSize);
        System.arraycopy(outputs, 0, outputsTrain, 0, trainSize);
        DataSet train = new DataSet(inputsTrain, outputsTrain);

        double[][] inputsTest = new double[testSize][];
        double[][] outputsTest = new double[testSize][];
        System.arraycopy(inputs, trainSize, inputsTest, 0, testSize);
        System.arraycopy(outputs, trainSize, outputsTest, 0, testSize);
        DataSet test = new DataSet(inputsTest, outputsTest);

        return new TrainTestSplit(train, test);
    }

    public void normalizeInputs(Normalizer normalizer) {
        normalizerInputs = normalizer;
        normalizerInputs.normalize(inputs);
    }

    public void normalizeOutputs(Normalizer normalizer) {
        normalizerOutputs = normalizer;
        normalizerOutputs.normalize(outputs);
    }

    public void denormalizeInputs() {
        normalizerInputs.denormalize(inputs);
    }

    public void denormalizeOutputs() {
        normalizerOutputs.denormalize(outputs);
    }

}
