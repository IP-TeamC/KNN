package de.fhdw.knn.run;

import java.io.IOException;

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
import de.fhdw.knn.trainer.learningrate.DecayLearningRate;
import de.fhdw.knn.trainer.loss.LossFunction;
import de.fhdw.knn.trainer.optimization.GradientDescent;
import de.fhdw.knn.trainer.optimization.OptimizationFunction;
import de.fhdw.knn.trainer.stop.EarlyStopping;
import de.fhdw.knn.trainer.stop.StopFunction;
import de.fhdw.knn.util.Chart;

public class SinusAdd {

    public static void main(String[] args) throws IOException {
        DataSet data = generate(10000, -10, 10);

        long trainStart = System.currentTimeMillis();
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.SNAKE, ActivationFunction.LINEAR, 1, 10,
                data.outputSize);
        ((DenseNeuron) denseLayers[0].neurons[0]).activationFunction = ActivationFunction.LINEAR;
        Network network = new Network(235890, WeightInitializer.GLOROT_UNIFORM, data.inputSize, denseLayers);
        // network =
        // Importer.importNetwork("models/sinusadd_lin_snake_lin_235890gr_1x1_mae_001dlr_10000_es1e-9-50.knn");
        // network = Importer.importNetwork("models/sinusadd_lin_sin_50gr_1x1_mae_001dlr_10000_es1e-9-50.knn");

        LossFunction lossFunction = LossFunction.MEAN_ABSOLUTE_ERROR;
        StopFunction stopFunction = new EarlyStopping(1e-9, 1000);
        OptimizationFunction optimizationFunction = new GradientDescent(lossFunction,
                new DecayLearningRate(0.0001, 0.995));

        Trainer trainer = new Trainer(network, 1, true, 1, lossFunction, stopFunction, optimizationFunction);
        // trainer.train(data);
        long trainStop = System.currentTimeMillis();

        long testStart = System.currentTimeMillis();
        DataSet testSet = generate(2503, -10000, 10000);
        // double[][] predictions = network.predict(testSet.inputs);
        // double totalLoss = lossFunction.totalLoss(testSet.outputs, predictions);
        // System.out.println("Total Loss: " + totalLoss);
        long testStop = System.currentTimeMillis();

        long trainTime = trainStop - trainStart;
        long testTime = testStop - testStart;
        System.out.printf("Train Time: %d ms (%.2f s)%n", trainTime, trainTime / 1000.0);
        System.out.printf("Test Time: %d ms (%.2f s)%n", testTime, testTime / 1000.0);

        double min = -1000;
        double max = 1000;
        double[][] inputsA = new double[100000][];
        double[][] inputsB = new double[inputsA.length][];
        double[][] inputs = new double[inputsA.length][];
        double step = (max - min) / inputsA.length;
        for (int i = 0; i < inputsA.length; i++) {
            inputsA[i] = new double[]{min + step * i, 0};
            inputsB[i] = new double[]{0, min + step * i};
            inputs[i] = new double[]{min + step * i};
        }

        // Chart.draw(network, inputsA, "a", 0, "sin(a+0)", 0);
        // Chart.draw(network, inputsB, "b", 1, "sin(0+b)", 0);

        Network add = Importer.importNetwork("models/add.knn");
        Network sin = Importer.importNetwork("models/sin_snake.knn");
        final Network addsin = new Network(235890, WeightInitializer.GLOROT_UNIFORM, data.inputSize,
                DenseLayer.createLayers(null, null, 0, 0, 0));
        addsin.denseLayers[0] = add.denseLayers[0];
        addsin.denseLayers[1] = add.denseLayers[1];
        addsin.denseLayers[2] = new DenseLayer(1);

        double[] adapterBias = new double[]{0};
        double[][] adapterWeights = new double[][]{new double[]{1}};
        SuperNeuron sn = new SuperNeuron(sin, adapterBias, adapterWeights);
        Connection conn = new Connection(1);
        conn.inputNeuron = add.denseLayers[1].neurons[0];
        sn.incoming = new Connection[]{conn};
        addsin.denseLayers[2].neurons[0] = sn;
        System.out.println("supi: " + sn);
        System.out.println(addsin.denseLayers[2].neurons[0] instanceof SuperNeuron);
        System.out.println("network: " + addsin);

        Chart.draw(addsin, inputsA, "a", 0, "sin(a+0)", 0);

        double[][] predictions = addsin.predict(testSet.inputs);
        double totalLoss = lossFunction.totalLoss(testSet.outputs, predictions);
        System.out.println("Total Loss: " + totalLoss);

        addsin.denseLayers[0].neurons[0].incoming[0].weight = 1.2694410904012163;// ;new Random().nextDouble(1.4);
        addsin.denseLayers[0].neurons[0].incoming[1].weight = 1.386981814728813;// new Random().nextDouble(1.4);

        for (DenseLayer layer : addsin.denseLayers) {
            for (AbstractDenseNeuron dn : layer.neurons) {
                System.out.print("DenseNeuron: " + (dn instanceof DenseNeuron rdn ? rdn.bias : "super") + " - ");
                for (Connection conn2 : dn.incoming) {
                    System.out.print(conn2.weight + " - ");
                }
                System.out.println();
            }
        }

        Trainer trainer2 = new Trainer(addsin, 100, true, 1, lossFunction, stopFunction, optimizationFunction);
        trainer2.train(data);

        for (DenseLayer layer : addsin.denseLayers) {
            for (AbstractDenseNeuron dn : layer.neurons) {
                System.out.print("DenseNeuron: " + (dn instanceof DenseNeuron rdn ? rdn.bias : "super") + " - ");
                for (Connection conn2 : dn.incoming) {
                    System.out.print(conn2.weight + " - ");
                }
                System.out.println();
            }
        }

        Chart.draw(addsin, inputs, "a", 0, "sin(a+0)", 0);
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
                outputs[count++] = new double[]{Math.sin(a + b)};
            }
        }
        return new DataSet(inputs, outputs);
    }

}
