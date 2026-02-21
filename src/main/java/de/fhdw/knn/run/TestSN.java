package de.fhdw.knn.run;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.io.Importer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.AbstractDenseNeuron;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.SuperNeuron;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.util.Chart;

import java.io.IOException;

public class TestSN {

    // Super-Neuron Test
    // Funktion 69*sin(3*x)+42x, wobei sin(x) das Super-Neuron ist (mit Snake-Aktivierungsfunktion trainiert)
    // Super-Neuron funktioniert ähnlich gut wie Sinus-Aktivierungsfunktion

    public static void main(String[] args) throws IOException {
        DataSet data = generate(10000, -10, 10);

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.LINEAR, 2, data.outputSize);
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, data.inputSize, denseLayers);
        new SuperNeuron(Importer.importNetwork("models/sin_snake.knn"), new double[]{0}, new double[][]{new double[]{1}}).insert(network, 0, 0);
        //((DenseNeuron)network.denseLayers[0].neurons[0]).activationFunction=ActivationFunction.SIN;

        //network.denseLayers[0].neurons[0].incoming[0].weight = 3.3;
        //network.denseLayers[0].neurons[1].incoming[0].weight = 1;
        //network.denseLayers[1].neurons[0].incoming[0].weight = 68;
        //network.denseLayers[1].neurons[0].incoming[1].weight = 43;

        for (DenseLayer layer : network.denseLayers) {
            for (AbstractDenseNeuron dn : layer.neurons) {
                System.out.print("DenseNeuron: " + (dn instanceof DenseNeuron rdn ? rdn.bias : "super") + " - ");
                for (Connection conn2 : dn.incoming) {
                    System.out.print(conn2.weight + " - ");
                }
                System.out.println();
            }
        }

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        StopFunction stopFunction = new EarlyStopping(1e-9, 5000);
        // kleine Learning Rate, sonst Exploding Gradient (weil Sinus periodisch?)
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new ConstantLearningRate(0.0000_0000_0_2));

        Trainer trainer = new Trainer(network, 3000, true, 1, lossFunction, stopFunction, optimizationFunction);
        trainer.train(data);

        long trainStop = System.currentTimeMillis();

        long testStart = System.currentTimeMillis();
        DataSet testSet = generate(2503, -100, 100);
        double[][] predictions = network.predict(testSet.inputs);
        double totalLoss = lossFunction.totalLoss(testSet.outputs, predictions);
        System.out.println("Total Loss: " + totalLoss);
        long testStop = System.currentTimeMillis();

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);

        double min = -30;
        double max = 30;
        double[][] inputs = new double[1000/*000*/][];
        double step = (max - min) / inputs.length;
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = new double[]{min + step * i};
        }

        Chart.draw(network, inputs, "x", 0, "sin(x)", 0);

        for (DenseLayer layer : network.denseLayers) {
            for (AbstractDenseNeuron dn : layer.neurons) {
                System.out.print("DenseNeuron: " + (dn instanceof DenseNeuron rdn ? rdn.bias : "super") + " - ");
                for (Connection conn2 : dn.incoming) {
                    System.out.print(conn2.weight + " - ");
                }
                System.out.println();
            }
        }
    }

    private static DataSet generate(int size, double min, double max) {
        double step = (max - min) / size;
        double[][] inputs = new double[size][];
        double[][] outputs = new double[size][];
        int count = 0;
        for (int i = 0; i < size; i++) {
            double x = min + step * i;
            inputs[count] = new double[]{x};
            double val = 69 * Math.sin(3 * x) + 42 * x;
            outputs[count++] = new double[]{val};
        }
        return new DataSet(inputs, outputs);
    }

}
