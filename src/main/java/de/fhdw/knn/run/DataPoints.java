package de.fhdw.knn.run;

import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.TrainTestSplit;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.io.Importer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.DecayLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import org.jfree.data.xy.XYSeries;

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
        network = Importer.importNetwork("datapoints.knn");

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        StopFunction stopFunction = EarlyStopping.NEVER;
        OptimizationFunction optimizationFunction = new GradientDescent(network, lossFunction, new DecayLearningRate(0.0003, 0.99));

        Trainer trainer = new Trainer(network, 50, true, 1, lossFunction, stopFunction, optimizationFunction);
        //trainer.train(train);
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

        XYSeries[] xys = new XYSeries[data.inputs[0].length];
        for (int column = 0; column < data.inputs[0].length; column++) {
            double min = findMin(data.inputs, column);
            double max = findMax(data.inputs, column);

            // funktionenschaar mit verändertem 2. parameter
            // alle funktionen/linien für alle datensätze und dann durchschnitt je x-wert

            data.shuffle(42);
            double[] base = data.inputs[0];
            double[][] inputs = new double[200][];
            double step = (max - min) / inputs.length;
            for (int i = 0; i < inputs.length; i++) {
                inputs[i] = new double[base.length];
                System.arraycopy(base, 0, inputs[i], 0, base.length);
                inputs[i][column] = step * i;
                System.out.println("i: " + i + ", x: " + inputs[i][column]);
            }

            double[] y = network.predictSingles(inputs);
            XYSeries series = new XYSeries("x" + column);
            for (int i = 0; i < inputs.length; i++) {
                series.add(inputs[i][column], y[i]);
                System.out.println("x: " + inputs[i][column] + ", y: " + y[i]);
            }
            xys[column] = series;
        }

        final ChartDemo demo = new ChartDemo("Chart Demo", xys);
        demo.pack();
        demo.setVisible(true);
    }

    public static double findMin(double[][] inputs, int column) {
        double min = Double.MAX_VALUE;
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i][column] < min) {
                min = inputs[i][column];
            }
        }
        return min;
    }

    public static double findMax(double[][] inputs, int column) {
        double max = Double.MIN_VALUE;
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i][column] > max) {
                max = inputs[i][column];
            }
        }
        return max;
    }

}
