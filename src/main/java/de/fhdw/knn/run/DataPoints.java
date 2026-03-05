package de.fhdw.knn.run;

import de.fhdw.knn.data.CsvReader;
import de.fhdw.knn.data.DataSet;
import de.fhdw.knn.data.MinMaxNormalizer;
import de.fhdw.knn.data.Normalizer;
import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.trainer.Trainer;
import de.fhdw.knn.trainer.learningrate.DecayLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import org.jfree.data.xy.XYSeries;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class DataPoints {

    public static void main(String[] args) throws IOException {
        DataSet data = CsvReader.readFile("data/datapoints.csv", 1, 10, 0, 1, 1);
        data.preprocess((input, output) -> output[0] /= input[0]);
        Normalizer normalizerInputs = new MinMaxNormalizer(-10, 10);
        Normalizer normalizerOutputs = new MinMaxNormalizer(-10, 10);
        data.normalizeInputs(normalizerInputs);
        data.normalizeInputs(normalizerOutputs);

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SNAKE, ActivationFunction.LINEAR, 5, 5, data.outputSize);
        for (DenseLayer layer : denseLayers) {
            for (int i = 0; i < Math.min(50, layer.neurons.length); i++) {
                DenseNeuron dn = (DenseNeuron) layer.neurons[i];
                if (dn.activationFunction == ActivationFunction.SNAKE) {
                    dn.activationFunction = ActivationFunction.SWISH;
                }
            }
        }
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, data.inputSize, denseLayers);
        //network = Importer.importNetwork("target/models/divx0_dp_swish_gr_3x200_mse_0001dlr_499.knn");

        LossFunction lossFunction = LossFunction.MEAN_SQUARED_ERROR;
        StopFunction stopFunction = EarlyStopping.NEVER;
        // learning rate zu klein oder decay zu groß (bzw. zu klein: näher an 1 - ist ja 1 - decay eig...)
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction, new DecayLearningRate(0.001, 0.995));

        Trainer trainer = new Trainer(network, 50, true, 1, lossFunction, stopFunction, optimizationFunction);
        //trainer.train(data, "target/models/divx0_dp_swishlt90_snake_gr_2x100_mse_0001dlr_%d.knn", 10);
        trainer.train(data);
        long trainStop = System.currentTimeMillis();

        network.export("models/datapoints_small5.knn");

        long testStart = System.currentTimeMillis();
        double[][] predictions = network.predict(data.inputs);
        double totalLoss = lossFunction.totalLoss(data.outputs, predictions);
        System.out.println("Total Loss: " + totalLoss);
        long testStop = System.currentTimeMillis();

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);


        int offset = 4500;
        int size = 1000;
        double[][] predictionInputs = new double[size][];
        System.arraycopy(data.inputs, offset, predictionInputs, 0, size);
        double[][] predictionOutputs = network.predict(predictionInputs);
        normalizerOutputs.denormalize(predictionOutputs);
        for (int i = 0; i < size; i++) {
            System.out.print("predicted: " + predictionOutputs[i][0] + " - ");
            System.out.println("expected: " + data.outputs[offset + i][0]);
        }
        //findAndPrintImpacts(network, data, 200);

        int count = 100;
        double a = 0;
        double b = 0;
        for (int counter = 0; counter < count; counter++) {
            List<XYSeries> xys = new LinkedList<>();
            double minX = Double.MAX_VALUE;
            double maxX = Double.MIN_VALUE;

            int baseIndex = new Random().nextInt(data.inputs.length);
            //int baseIndex = counter;
            double[] base = data.inputs[baseIndex];
            double origy = data.outputs[baseIndex][0];
            double orig0 = base[0];
            double orig1 = base[1];
            double orig2 = base[2];

            for (int column = 0; column < data.inputs[0].length; column++) {
                //if (!List.of(0, 1, 2).contains(column)) continue;
                double min = findMin(data.inputs, column);
                double max = findMax(data.inputs, column);

                // funktionenschaar mit verändertem 2. parameter
                // alle funktionen/linien für alle datensätze und dann durchschnitt je x-wert

                //x0 mit konstantem faktor
                //
                //x0 linear positive (und x0 ist faktor für alles, weil x0=0 => y ist null)
                //x1 wurzel funktion mit faktor (darin x2 enthalten) (eher kein logarithmus)
                //x2 linear negativ oder exponentiell?
                // x2 bestimmt steigung von x1-anteil (positiv negativ und größe)
                // x0-x0/x2*sqrt(x1)
                // x0*e^(-x2)
                // Partielle Ableitung nach x0 ist konstant
                // Partielle Ableitung nach x1 ist
                // Partielle Ableitung nach x2 ist negativ, x2 exponentiell

                data.shuffle(42);

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

                double localA = 0;
                double localB = 0;

                double[] y = network.predictSingles(inputs);
                XYSeries series = new XYSeries("x" + column);
                for (int i = 0; i < inputs.length; i++) {
                    if (i == 0) {
//                        System.out.print("orig y: " + origy + ", ");
//                        System.out.print("orig x" + 0 + ": " + orig0 + ", ");
//                        System.out.print("orig x" + 1 + ": " + orig1 + ", ");
//                        System.out.println("orig x" + 2 + ": " + orig2);
                    } else if (column == 0) {
                        // x0 linear
                        localA += y[i] / inputs[i][column];
                        //System.out.println(y[i] / inputs[i][column]);
                        //System.out.println("qy: " + y[i] / y[i - 1]);
                    } else if (column == 1) {
                        //System.out.println("qy: " + y[i] / y[i - 1]);
                        double fourX1 = 4 * inputs[i][column];
                        if (fourX1 <= maxX && false) {
                            double[] fourX1input = new double[base.length];
                            System.arraycopy(base, 0, fourX1input, 0, base.length);
                            fourX1input[column] = fourX1;
                            double twice = network.predict(new double[][]{fourX1input})[0][0];
                            System.out.println((twice - y[0]) / (y[i] - y[0]));
                        }
                    } else if (column == 2) {
                        double prevY = y[i - 1];
                        if (prevY == 0) {
                            prevY = 1e-6;
                        }
                        double div = y[i] / prevY;
                        if (div <= 0) {
                            div = 1e-6;
                        }
                        double log = Math.log(div);
                        if (Double.isNaN(y[i] / prevY)) {
                            throw new RuntimeException("div nan?");
                        }
                        if (Double.isNaN(log)) {
                            throw new RuntimeException("log nan?");
                        }
                        double bDiff = log / (inputs[i][column] - inputs[i - 1][column]);
                        //System.out.println(bDiff);
                        if (inputs[i][column] == inputs[i - 1][column]) {
                            throw new RuntimeException(inputs[i][column] + " - " + inputs[i - 1][column]);
                        } else if (inputs[i][column] - inputs[i - 1][column] == 0) {
                            throw new RuntimeException("x diff zero");
                        }
                        if (Double.isNaN(bDiff)) {
                            throw new RuntimeException("bDiff nan?");
                        }
                        localB += bDiff;
                        //System.out.println("qy: " + y[i] / y[i - 1]);
                        // x2 exponentiell fallend
                        //System.out.println("qy: " + Math.log(y[i] / y[i - 1]) / (inputs[i][column] / inputs[i - 1][column]));
                    }
                    series.add(inputs[i][column], y[i]);
                    //System.out.println("x: " + inputs[i][column] + ", y: " + y[i]);
                }
                if (column == 0) {
                    localA /= (inputs.length - 1);
                    //System.out.println("Local a: " + localA);
                    a += localA;
                }
                if (column == 2) {
                    localB /= (inputs.length - 1);
                    //System.out.println("Local b: " + localB);
                    b += localB;
                }
                xys.add(series);
            }

            LineChart chart = new LineChart("Line Chart", "x", "y", minX, maxX, xys.toArray(XYSeries[]::new));
            try {
                if (true)
                    Thread.sleep(5000);
                chart.setVisible(false);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        a /= count;
        b /= count;
        System.out.println("a: " + a);
        System.out.println("b: " + b);
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
