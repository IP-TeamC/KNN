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

public class SinusAdd {

    public static void main(String[] args) throws IOException {
        DataSet data = generate(10000, -10, 10);

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SNAKE, ActivationFunction.LINEAR, 1, 10, data.outputSize);
        denseLayers[0].neurons[0].activationFunction = ActivationFunction.LINEAR;
        Network network = new Network(235890, WeightInitializer.GLOROT_UNIFORM, data.inputSize, denseLayers);
        //network = Importer.importNetwork("sinusadd_lin_snake_lin_235890gr_1x1_mae_001dlr_10000_es1e-9-50.knn");
        //network = Importer.importNetwork("sinusadd_lin_sin_50gr_1x1_mae_001dlr_10000_es1e-9-50.knn");

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

        double min = -10000;
        double max = 10000;
        double[][] inputsA = new double[100000][];
        double[][] inputsB = new double[inputsA.length][];
        double step = (max - min) / inputsA.length;
        for (int i = 0; i < inputsA.length; i++) {
            inputsA[i] = new double[]{min + step * i, 0};
            inputsB[i] = new double[]{0, min + step * i};
        }

        Chart.draw(network, inputsA, "a", 0, "sin(a+0)", 0);
        Chart.draw(network, inputsB, "b", 1, "sin(0+b)", 0);
    }

    private static DataSet generate(int size, double min, double max) {
        int sizeSqrt = (int) Math.sqrt(size);
        double step = (max - min) / sizeSqrt;
        double[][] inputs = new double[sizeSqrt * sizeSqrt][];
        double[][] outputs = new double[sizeSqrt * sizeSqrt][];
        int count = 0;
        for (int i = 0; i < sizeSqrt; i++) {
            double a = min + step * i;
            for (int j = 0; j < sizeSqrt; j++) {
                double b = min + step * j;
                inputs[count] = new double[]{a, b};
                outputs[count++] = new double[]{Math.cos(a + b)};
            }
        }
        return new DataSet(inputs, outputs);
    }

}
