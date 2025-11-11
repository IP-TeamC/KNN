package de.fhdw.knn.run;

import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.TrainTestSplit;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;

import java.io.IOException;

public class DataPoints {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("datapoints.csv", 1, 10, 0, 1, 1);
        TrainTestSplit trainTest = data.shuffleAndSplit(42, 0.2);
        DataSet train = trainTest.train;
        DataSet test = trainTest.test;

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.RELU, ActivationFunction.LINEAR, 100, 100, 1);
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 10, denseLayers);
        //network = Importer.importNetwork("bq.knn");

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        StopFunction stopFunction = EarlyStopping.NEVER;
        OptimizationFunction optimizationFunction = new GradientDescent(network, lossFunction, new ConstantLearningRate(0.0001));

        Trainer trainer = new Trainer(network, 100, true, 1, lossFunction, stopFunction, optimizationFunction);
        trainer.train(train);
        long trainStop = System.currentTimeMillis();

        long testStart = System.currentTimeMillis();
        double[][] predictions = network.predict(test.inputs);
        double totalLoss = lossFunction.totalLoss(test.outputs, predictions);
        System.out.println("Total Loss: " + totalLoss);
        long testStop = System.currentTimeMillis();

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);
    }

}
