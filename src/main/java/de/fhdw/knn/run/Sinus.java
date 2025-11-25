package de.fhdw.knn.run;

import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.DecayLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.util.Chart;

import java.io.IOException;

public class Sinus {

    public static void main(String[] args) throws IOException {
        DataSet data = generate(10000, -10, 10);

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SIN, ActivationFunction.LINEAR, 1, data.outputSize);
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, data.inputSize, denseLayers);
        //network = Importer.importNetwork("models/sin_snake.knn");
        //network = Importer.importNetwork("models/sin_sin.knn");

        LossFunction lossFunction = LossFunction.MEAN_ABSOLUTE_ERROR;
        StopFunction stopFunction = new EarlyStopping(1e-9, 50);
        OptimizationFunction optimizationFunction = new GradientDescent(network, lossFunction, new DecayLearningRate(0.001, 0.995));

        Trainer trainer = new Trainer(network, 10000, true, 1, lossFunction, stopFunction, optimizationFunction);
        trainer.train(data);
        long trainStop = System.currentTimeMillis();

        long testStart = System.currentTimeMillis();
        DataSet testSet = generate(2503, -10000, 10000);
        double[][] predictions = network.predict(testSet.inputs);
        double totalLoss = lossFunction.totalLoss(testSet.outputs, predictions);
        System.out.println("Total Loss: " + totalLoss);
        long testStop = System.currentTimeMillis();

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);

        double min = 0;
        double max = 100000;
        double[][] inputs = new double[1000000][];
        double step = (max - min) / inputs.length;
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = new double[]{min + step * i};
        }

        Chart.draw(network, inputs, "x", 0, "sin(x)", 0);
    }

    private static DataSet generate(int size, double min, double max) {
        double step = (max - min) / size;
        double[][] inputs = new double[size][];
        double[][] outputs = new double[size][];
        int count = 0;
        for (int i = 0; i < size; i++) {
            double x = min + step * i;
            inputs[count] = new double[]{x};
            outputs[count++] = new double[]{Math.sin(x)};
        }
        return new DataSet(inputs, outputs);
    }

}
