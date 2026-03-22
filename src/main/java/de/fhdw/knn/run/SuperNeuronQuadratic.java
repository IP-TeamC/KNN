package de.fhdw.knn.run;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.io.Importer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.SuperNeuron;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.StopFunction;

import java.io.IOException;

class SuperNeuronQuadratic {

    // Netzwerk zur beispielhaften Verwendung des Super-Neurons
    // benutzt das Super-Neuron für ein vortrainiertes Netzwerk für eine quadratische Funktion
    // benutzt ein zusätzliches Neuron für das Training einer linearen Funktion
    // Ziel: 0.5*x^2 + 10x lernen
    // Hinweis: Super-Neuronen sind sehr anfällig für Exploding Gradients und setzen deshalb eine kleine Learning Rate voraus
    // hier ist möglicherweise das Ausprobieren verschiedener Trainings-Parameter notwendig

    public static void main(String[] args) throws IOException {
        DataSet data = generate(10000, -10, 10);

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 2, data.outputSize);
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, data.inputSize, denseLayers);
        // quadratische Funktion trainiert für Inputs zwischen -10 und 10
        new SuperNeuron(Importer.importNetwork("models/f2.knn"), new double[]{0}, new double[][]{new double[]{1}}).insert(network, 0, 0);

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        StopFunction stopFunction = StopFunction.NEVER;
        // kleine Learning Rate, sonst Exploding Gradient
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(0.0000_00_8));

        Trainer trainer = new Trainer(network, 100, true, 64, lossFunction, stopFunction, optimizationFunction);
        trainer.train(data);
        long trainStop = System.currentTimeMillis();

        long testStart = System.currentTimeMillis();
        double[][] predictions = network.predict(data.inputs);
        double totalLoss = lossFunction.totalLoss(data.outputs, predictions);
        System.out.println("Total Loss: " + totalLoss);
        long testStop = System.currentTimeMillis();

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);

        Chart.draw(network, data.inputs, "x", 0, "0.5*x^2 + 10x", 0);
    }

    private static DataSet generate(int size, double min, double max) {
        double step = (max - min) / size;
        double[][] inputs = new double[size][];
        double[][] outputs = new double[size][];
        int count = 0;
        for (int i = 0; i < size; i++) {
            double x = min + step * i;
            inputs[count] = new double[]{x};
            double val = 0.5 * x * x + 10 * x;
            outputs[count++] = new double[]{val};
        }
        return new DataSet(inputs, outputs);
    }

}
