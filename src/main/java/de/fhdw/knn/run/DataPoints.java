package de.fhdw.knn.run;

import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
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

import javax.xml.crypto.Data;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

public class DataPoints {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("datapoints.csv", 1, 3, 0, 1, 1);

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SWISH, ActivationFunction.LINEAR, 200, 200, 200, data.outputSize);
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, data.inputSize, denseLayers);
        network = Importer.importNetwork("x012_dp_swish_gr_3x200_mae_0001dlr_499.knn");

        LossFunction lossFunction = LossFunction.MEAN_ABSOLUTE_ERROR;
        StopFunction stopFunction = EarlyStopping.NEVER;
        OptimizationFunction optimizationFunction = new GradientDescent(network, lossFunction, new DecayLearningRate(0.0001, 0.99));

        Trainer trainer = new Trainer(network, 500, true, 1, lossFunction, stopFunction, optimizationFunction);
        //trainer.train(data);
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

        for (int i = 0; i < 50; i++) {
            System.out.print("predicted: " + network.predictSingles(data.inputs[i])[0] + " - ");
            System.out.println("expected: " + data.outputs[i][0]);
        }

        if (true)
            return;
        while (true) {
            List<XYSeries> xys = new LinkedList<>();
            double minX = Double.MAX_VALUE;
            double maxX = Double.MIN_VALUE;
            for (int column = 0; column < data.inputs[0].length; column++) {
                if (!List.of(0, 1, 2, 8).contains(column)) continue;
                double min = findMin(data.inputs, column);
                double max = findMax(data.inputs, column);

                // funktionenschaar mit verändertem 2. parameter
                // alle funktionen/linien für alle datensätze und dann durchschnitt je x-wert

                //x0 mit konstantem faktor
                //
                //x0 linear positive (und x0 ist faktor für alles, weil x0=0 => y ist null)
                //x1 wurzel funktion mit faktor (darin x2 enthalten) (eher kein logarithmus)
                //x2 linear negativ oder exponentiell?
                //
                //(a*x0) + (b*x0)()

                data.shuffle(42);
                double[] base = data.inputs[new Random().nextInt(data.inputs.length)];
                double[][] inputs = new double[200][];
                double step = (max - min) / inputs.length;
                for (int i = 0; i < inputs.length; i++) {
                    inputs[i] = new double[base.length];
                    System.arraycopy(base, 0, inputs[i], 0, base.length);
                    inputs[i][column] = min + step * i;
                    if (inputs[i][column] > maxX) maxX = inputs[i][column];
                    if (inputs[i][column] < minX) minX = inputs[i][column];
                    //System.out.println("i: " + i + ", x: " + inputs[i][column]);
                }

                double[] y = network.predictSingles(inputs);
                XYSeries series = new XYSeries("x" + column);
                for (int i = 0; i < inputs.length; i++) {
                    series.add(inputs[i][column], y[i]);
                    //System.out.println("x: " + inputs[i][column] + ", y: " + y[i]);
                }
                xys.add(series);
            }

            LineChart chart = new LineChart("Line Chart", "x", "y", minX, maxX, xys.toArray(XYSeries[]::new));
            try {
                Thread.sleep(6000);
                chart.setVisible(false);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private static void findAndPrintImpacts(Network network, DataSet data, int steps) {
        double[] impacts = findImpacts(network, data, steps);
        for (int column = 0; column < data.inputSize; column++) {
            System.out.println("Impact x" + column + ": " + impacts[column]);
        }
    }

    static volatile int counter = 0;
    public static double[] findImpacts(Network network, DataSet data, int steps) {
        double[] impact = new double[data.inputSize];
        double[] min = new double[data.inputSize];
        double[] step = new double[data.inputSize];

        for (int column = 0; column < data.inputSize; column++) {
            min[column] = findMin(data.inputs, column);
            step[column] = (findMax(data.inputs, column) - min[column]) / steps;
        }

        IntStream.range(0, data.inputs.length).parallel().forEach(i -> {
            double[] minPred = new double[data.inputSize];
            double[] maxPred = new double[data.inputSize];

            for (int column = 0; column < data.inputSize; column++) {
                minPred[column] = Double.MAX_VALUE;
                maxPred[column] = Double.MIN_VALUE;
                double[] base = data.inputs[i];
                for (int j = 0; j < steps; j++) {
                    base[column] = min[column] + step[column] * j;
                    double pred = network.predict(new double[][]{base})[0][0];
                    if (pred > maxPred[column]) maxPred[column] = pred;
                    if (pred < minPred[column]) minPred[column] = pred;
                }
            }

            double[] ranges = IntStream.range(0, data.inputSize).mapToDouble(column -> maxPred[column] - minPred[column]).toArray();
            double sumRanges = Arrays.stream(ranges).sum();
            synchronized (impact) {
                for (int column = 0; column < data.inputSize; column++) {
                    impact[column] += ranges[column] / sumRanges;
                }
                counter++;
            }
            System.out.println(counter);
        });

        for (int column = 0; column < data.inputSize; column++) {
            impact[column] /= data.inputs.length;
        }
        return impact;
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
