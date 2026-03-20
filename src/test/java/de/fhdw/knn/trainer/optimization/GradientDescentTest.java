package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.util.TestUtil;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.OutputsDerived;
import de.fhdw.knn.network.neuron.SuperNeuron;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.loss.LossFunction;

import javax.annotation.processing.Generated;
import java.util.Arrays;

public class GradientDescentTest {

    @Test
    public void testAlmostPerfect() {
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.SIN, 1, 1));
        OptimizationFunction gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.1));
        gradientDescent.epoch(0, Double.NaN);
        network.denseLayers[0].neurons[0].incoming[0].weight = 42;
        network.denseLayers[0].neurons[0].incoming[1].weight = 24;
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.25;

        Adjustments adjustments = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adjustments, buffer, network, new double[]{-3, 7}, new double[]{-0.87969576}, 1);
        Arrays.stream(adjustments.bias).flatMapToDouble(Arrays::stream)
                .forEach(adjustmentBias -> assertEquals(0.0, adjustmentBias, 1e-8));
        Arrays.stream(adjustments.weight).flatMap(Arrays::stream).flatMapToDouble(Arrays::stream)
                .forEach(adjustmentWeight -> assertEquals(0.0, adjustmentWeight, 1e-8));
    }

    @Test
    public void testLearningRate() {
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.SIN, 1, 1));

        OptimizationFunction gradientDescentSmall = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.1));
        gradientDescentSmall.epoch(0, Double.NaN);

        OptimizationFunction gradientDescentBig = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(1));
        gradientDescentBig.epoch(0, Double.NaN);

        network.denseLayers[0].neurons[0].incoming[0].weight = 41;
        network.denseLayers[0].neurons[0].incoming[1].weight = 25;
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.28;

        Adjustments adjustmentsSmall = Adjustments.generateEmpty(network);
        Adjustments adjustmentsBig = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescentSmall.compute(adjustmentsSmall, buffer, network, new double[]{-3, 7}, new double[]{-0.87969576}, 1);
        gradientDescentBig.compute(adjustmentsBig, buffer, network, new double[]{-3, 7}, new double[]{-0.87969576}, 1);

        for (int layer = 0; layer < network.denseLayers.length; layer++) {
            for (int neuron = 0; neuron < network.denseLayers[layer].neurons.length; neuron++) {
                assertEquals(10,
                        adjustmentsBig.bias[layer][neuron] / adjustmentsSmall.bias[layer][neuron], 1e-10);
                for (int conn = 0; conn < network.denseLayers[layer].neurons[neuron].incoming.length; conn++) {
                    assertEquals(10,
                            adjustmentsBig.weight[layer][neuron][conn] / adjustmentsSmall.weight[layer][neuron][conn], 1e-10);
                }
            }
        }
    }

    @Test
    public void testAdjustmentsLargeNegative() {
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.TANH, 1, 1));

        OptimizationFunction gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(1));
        gradientDescent.epoch(0, Double.NaN);

        network.denseLayers[0].neurons[0].incoming[0].weight = 42;
        network.denseLayers[0].neurons[0].incoming[1].weight = 24;
        // sollte 0.25 sein (-> Adjustment/Gradient soll groß negativ sein, um das Gewicht zu erhöhen - Anpassung wird immer subtrahiert)
        network.denseLayers[1].neurons[0].incoming[0].weight = -0.5;

        Adjustments adjustments = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adjustments, buffer, network, new double[]{-3, 5.4}, new double[]{0.71629787}, 1);
        assertTrue(adjustments.weight[1][0][0] < -1);
    }

    @Test
    public void testAdjustmentsPositive() {
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.TANH, 1, 1));

        OptimizationFunction gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(1));
        gradientDescent.epoch(0, Double.NaN);

        network.denseLayers[0].neurons[0].incoming[0].weight = 42;
        network.denseLayers[0].neurons[0].incoming[1].weight = 24;
        // sollte 0.25 sein (-> Adjustment/Gradient soll positiv sein, um das Gewicht zu erhöhen - Anpassung wird immer subtrahiert)
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.4;

        Adjustments adjustments = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adjustments, buffer, network, new double[]{-3, 5.4}, new double[]{0.71629787}, 1);
        assertTrue(adjustments.weight[1][0][0] > 0.2);
    }

    @Test
    public void testAdjustmentsDifferenceQuotientWeights() {
        // Verwendung der h-Methode / des Differenzialquotienten zur Bestimmung der Ableitung/des Gradienten
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.TANH, 1, 1));

        LossFunction mse = LossFunction.MEAN_SQUARED_ERROR;
        OptimizationFunction gradientDescent = new GradientDescent(mse, new ConstantLearningRate(1));
        gradientDescent.epoch(0, Double.NaN);

        network.denseLayers[0].neurons[0].incoming[0].weight = 42;
        network.denseLayers[0].neurons[0].incoming[1].weight = 24;

        double[] input = new double[]{-3, 5.4};
        double[] output = new double[]{0.71629787};

        // Gewicht im Output Layer
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.30001;
        double[] prediction1 = network.predict(new double[][]{input})[0];
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.30002;
        double[] prediction2 = network.predict(new double[][]{input})[0];

        double loss1 = mse.loss(output, prediction1);
        double loss2 = mse.loss(output, prediction2);
        double gradient12 = (loss2 - loss1) / 0.00001;
        assertNotEquals(0, gradient12, 1e-4);

        network.denseLayers[1].neurons[0].incoming[0].weight = 0.300015;
        Adjustments adjustments12 = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adjustments12, buffer, network, input, output, 1);
        assertEquals(gradient12, adjustments12.weight[1][0][0], 1e-8);

        // Gewicht im Hidden Layer
        network.denseLayers[0].neurons[0].incoming[0].weight = 41.50003;
        double[] prediction3 = network.predict(new double[][]{input})[0];
        network.denseLayers[0].neurons[0].incoming[0].weight = 41.50004;
        double[] prediction4 = network.predict(new double[][]{input})[0];

        double loss3 = mse.loss(output, prediction3);
        double loss4 = mse.loss(output, prediction4);
        double gradient34 = (loss4 - loss3) / 0.00001;
        assertNotEquals(0, gradient34, 1e-4);

        network.denseLayers[0].neurons[0].incoming[0].weight = 41.500035;
        Adjustments adjustments34 = Adjustments.generateEmpty(network);
        gradientDescent.compute(adjustments34, buffer, network, input, output, 1);
        assertEquals(gradient34, adjustments34.weight[0][0][0], 1e-8);
    }

    @Test
    public void testAdjustmentsDifferenceQuotientBias() {
        // Verwendung der h-Methode / des Differenzialquotienten zur Bestimmung der Ableitung/des Gradienten
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.TANH, 1, 1));

        LossFunction mse = LossFunction.MEAN_SQUARED_ERROR;
        OptimizationFunction gradientDescent = new GradientDescent(mse, new ConstantLearningRate(1));
        gradientDescent.epoch(0, Double.NaN);

        network.denseLayers[0].neurons[0].incoming[0].weight = 42;
        network.denseLayers[0].neurons[0].incoming[1].weight = 24;
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.25;

        double[] input = new double[]{-3, 5.4};
        double[] output = new double[]{0.71629787};

        // Gewicht im Output Layer
        ((DenseNeuron) network.denseLayers[1].neurons[0]).bias = 0.20001;
        double[] prediction1 = network.predict(new double[][]{input})[0];
        ((DenseNeuron) network.denseLayers[1].neurons[0]).bias = 0.20002;
        double[] prediction2 = network.predict(new double[][]{input})[0];

        double loss1 = mse.loss(output, prediction1);
        double loss2 = mse.loss(output, prediction2);
        double gradient12 = (loss2 - loss1) / 0.00001;
        assertNotEquals(0, gradient12, 1e-4);

        ((DenseNeuron) network.denseLayers[1].neurons[0]).bias = 0.200015;
        Adjustments adjustments12 = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adjustments12, buffer, network, input, output, 1);
        assertEquals(gradient12, adjustments12.bias[1][0], 1e-8);

        // Gewicht im Hidden Layer
        ((DenseNeuron) network.denseLayers[0].neurons[0]).bias = 0.15003;
        double[] prediction3 = network.predict(new double[][]{input})[0];
        ((DenseNeuron) network.denseLayers[0].neurons[0]).bias = 0.15004;
        double[] prediction4 = network.predict(new double[][]{input})[0];

        double loss3 = mse.loss(output, prediction3);
        double loss4 = mse.loss(output, prediction4);
        double gradient34 = (loss4 - loss3) / 0.00001;
        assertNotEquals(0, gradient34, 1e-4);

        ((DenseNeuron) network.denseLayers[0].neurons[0]).bias = 0.150035;
        Adjustments adjustments34 = Adjustments.generateEmpty(network);
        gradientDescent.compute(adjustments34, buffer, network, input, output, 1);
        assertEquals(gradient34, adjustments34.bias[0][0], 1e-8);
    }

    @Test
    public void testAdjustmentsDifferenceQuotientWeightsSuperNeuron() {
        // Verwendung der h-Methode / des Differenzialquotienten zur Bestimmung der Ableitung/des Gradienten
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.TANH, 1, 1));
        // Verhalten wie eigentlich definiertes Netzwerk über Dummy-SuperNeuron
        new SuperNeuron(TestUtil.simpleDummyNetwork(), new double[]{0}, new double[][]{new double[]{1, 1}}).insert(network, 0, 0);
        Network tanh = TestUtil.simpleDummyNetwork();
        ((DenseNeuron) tanh.denseLayers[0].neurons[0]).activationFunction = ActivationFunction.TANH;
        new SuperNeuron(tanh).insert(network, 1, 0);

        LossFunction mse = LossFunction.MEAN_SQUARED_ERROR;
        OptimizationFunction gradientDescent = new GradientDescent(mse, new ConstantLearningRate(1));
        gradientDescent.epoch(0, Double.NaN);

        network.denseLayers[0].neurons[0].incoming[0].weight = 42;
        network.denseLayers[0].neurons[0].incoming[1].weight = 24;

        double[] input = new double[]{-3, 5.4};
        double[] output = new double[]{0.71629787};

        // Gewicht im Output Layer
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.30001;
        double[] prediction1 = network.predict(new double[][]{input})[0];
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.30002;
        double[] prediction2 = network.predict(new double[][]{input})[0];

        double loss1 = mse.loss(output, prediction1);
        double loss2 = mse.loss(output, prediction2);
        double gradient12 = (loss2 - loss1) / 0.00001;
        assertNotEquals(0, gradient12, 1e-4);

        network.denseLayers[1].neurons[0].incoming[0].weight = 0.300015;
        Adjustments adjustments12 = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adjustments12, buffer, network, input, output, 1);
        assertEquals(gradient12, adjustments12.weight[1][0][0], 1e-8);

        // Gewicht im Hidden Layer
        network.denseLayers[0].neurons[0].incoming[0].weight = 41.50003;
        double[] prediction3 = network.predict(new double[][]{input})[0];
        network.denseLayers[0].neurons[0].incoming[0].weight = 41.50004;
        double[] prediction4 = network.predict(new double[][]{input})[0];

        double loss3 = mse.loss(output, prediction3);
        double loss4 = mse.loss(output, prediction4);
        double gradient34 = (loss4 - loss3) / 0.00001;
        assertNotEquals(0, gradient34, 1e-4);

        network.denseLayers[0].neurons[0].incoming[0].weight = 41.500035;
        Adjustments adjustments34 = Adjustments.generateEmpty(network);
        gradientDescent.compute(adjustments34, buffer, network, input, output, 1);
        assertEquals(gradient34, adjustments34.weight[0][0][0], 1e-8);
    }

    @Test
    public void testAdjustmentsDifferenceQuotientWeightsSuperNeuron3DenseLayers() {
        // Verwendung der h-Methode / des Differenzialquotienten zur Bestimmung der Ableitung/des Gradienten
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.TANH, 1, 1, 1));
        new SuperNeuron(TestUtil.simpleDummyNetwork(), new double[]{0}, new double[][]{new double[]{1, 1}}).insert(network, 0, 0);

        LossFunction mse = LossFunction.MEAN_SQUARED_ERROR;
        OptimizationFunction gradientDescent = new GradientDescent(mse, new ConstantLearningRate(1));
        gradientDescent.epoch(0, Double.NaN);

        network.denseLayers[0].neurons[0].incoming[0].weight = 42;
        network.denseLayers[0].neurons[0].incoming[1].weight = 24;
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.5;
        network.denseLayers[2].neurons[0].incoming[0].weight = 0.5;

        double[] input = new double[]{-3, 5.4};
        double[] output = new double[]{0.71629787};

        // Gewicht im 2. Hidden Layer
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.50001;
        double[] prediction1 = network.predict(new double[][]{input})[0];
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.50002;
        double[] prediction2 = network.predict(new double[][]{input})[0];

        double loss1 = mse.loss(output, prediction1);
        double loss2 = mse.loss(output, prediction2);
        double gradient12 = (loss2 - loss1) / 0.00001;
        assertNotEquals(0, gradient12, 1e-4);

        network.denseLayers[1].neurons[0].incoming[0].weight = 0.500015;
        Adjustments adjustments12 = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adjustments12, buffer, network, input, output, 1);
        assertEquals(gradient12, adjustments12.weight[1][0][0], 1e-8);
    }

    // ===== 1. EINFACHE STRUKTUR-TESTS =====

    @Generated("GitHub Copilot")
    @Test
    public void testSingleNeuronGradientComputation() {
        // 1 Input -> 2 Hidden -> 1 Output
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {1.0};
        double[] expectedOutput = {1.0};

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expectedOutput, 1);

        assertNotNull(adj.weight);
        assertNotNull(adj.bias);
        assertEquals(2, adj.weight.length);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testSingleNeuronGradientComputationSuperNeurons() {
        // 1 Input -> 1 Hidden -> 1 Output
        Network network = createNetworkWithHiddenLayer(1, 1, 1);
        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );
        new SuperNeuron(TestUtil.simpleDummyNetwork()).insert(network, 0, 0);
        new SuperNeuron(TestUtil.simpleDummyNetwork()).insert(network, 1, 0);

        double[] input = {1.0};
        double[] expectedOutput = {1.0};

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expectedOutput, 1);

        assertNotNull(adj.weight);
        assertNotNull(adj.bias);
        assertEquals(2, adj.weight.length);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testSingleNeuronGradientComputationSuperNeuronsSimpler() {
        // 1 Input -> 1 output
        Network network = TestUtil.simpleDummyNetwork();
        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );
        new SuperNeuron(TestUtil.simpleDummyNetwork()).insert(network, 0, 0);

        double[] input = {1.0};
        double[] expectedOutput = {1.0};

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expectedOutput, 1);

        assertNotNull(adj.weight);
        assertNotNull(adj.bias);
        assertEquals(1, adj.weight.length);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testMultipleOutputNeurons() {
        // 1 Input -> 3 Hidden -> 3 Output
        Network network = createNetworkWithHiddenLayer(1, 3, 3);
        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5};
        double[] expectedOutput = {0.8, 0.2, 0.5};

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expectedOutput, 1);

        assertEquals(2, adj.weight.length);
        assertEquals(2, adj.bias.length);
        assertEquals(3, adj.weight[1].length);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testTwoLayerNetworkGradients() {
        // 2 Input -> 3 Hidden -> 1 Output
        Network network = createNetworkWithMultipleHiddenLayers(2, new int[]{3, 2}, 1);
        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5, -0.3};
        double[] expectedOutput = {0.8};

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expectedOutput, 1);

        // 3 Layer total: Input -> 3 Hidden -> 2 Hidden -> 1 Output
        assertEquals(3, adj.weight.length);
        assertEquals(3, adj.bias.length);
    }

    // ===== 2. GRADIENT-EIGENSCHAFTEN =====

    @Generated("GitHub Copilot")
    @Test
    public void testGradientProportionalToLearningRate() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);

        Adjustments adj1 = computeWithLearningRate(network, 0.01);
        Adjustments adj2 = computeWithLearningRate(network, 0.02);

        // Bei 2x Lernrate sollten Gradienten ~2x sein
        if (Math.abs(adj1.bias[1][0]) > 1e-10) {
            double ratio = adj2.bias[1][0] / adj1.bias[1][0];
            assertEquals(2.0, ratio, 0.05);
        }
    }

    @Generated("GitHub Copilot")
    @Test
    public void testGradientInverselyProportionalToBatchSize() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] input = {1.0};
        double[] output = {0.5};

        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adjBatch1 = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adjBatch1, buffer, network, input, output, 1);

        gradientDescent.epoch(0, 0);
        Adjustments adjBatch32 = Adjustments.generateEmpty(network);
        gradientDescent.compute(adjBatch32, buffer, network, input, output, 32);

        // Mit 32x Batch sollten Adjustments 1/32 sein
        if (Math.abs(adjBatch32.bias[1][0]) > 1e-10) {
            double ratio = adjBatch1.bias[1][0] / adjBatch32.bias[1][0];
            assertEquals(32.0, ratio, 1.0);
        }
    }

    // ===== 3. SPEZIALFÄLLE =====

    @Generated("GitHub Copilot")
    @Test
    public void testZeroLearningRateNoChanges() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] input = {1.0};
        double[] output = {0.5};

        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.0)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, output, 1);

        assertTrue(allAdjustmentsZero(adj));
    }

    @Generated("GitHub Copilot")
    @Test
    public void testLargeInputsStability() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] largeInput = {1000.0};
        double[] output = {1.0};

        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, largeInput, output, 1);

        assertFalse(containsNaN(adj), "Gradient contains NaN values");
        assertFalse(containsInfinity(adj), "Gradient contains Infinity values");
    }

    @Generated("GitHub Copilot")
    @Test
    public void testVerySmallLearningRateStability() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] input = {1.0};
        double[] output = {0.5};

        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(1e-8)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, output, 1);

        assertFalse(containsNaN(adj), "Gradient with tiny learning rate contains NaN");
        assertFalse(containsInfinity(adj), "Gradient with tiny learning rate contains Infinity");
    }

    @Generated("GitHub Copilot")
    @Test
    public void testNegativeInputsStability() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] negativeInput = {-100.0};
        double[] output = {0.0};

        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, negativeInput, output, 1);

        assertFalse(containsNaN(adj));
        assertFalse(containsInfinity(adj));
    }

    // ===== 4. BACKPROPAGATION KONSISTENZ =====

    @Generated("GitHub Copilot")
    @Test
    public void testMultiLayerGradientFlow() {
        // 2 Input -> 3 Hidden -> 3 Output
        Network network = createNetworkWithHiddenLayer(2, 3, 3);
        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5, -0.3};
        double[] expectedOutput = {0.8, 0.2, 0.5};

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expectedOutput, 1);

        // Prüfe dass beide Layer Gradienten haben
        assertEquals(2, adj.weight.length);
        assertNotNull(adj.weight[0], "Hidden layer weight adjustments null");
        assertNotNull(adj.weight[1], "Output layer weight adjustments null");
        assertNotNull(adj.bias[0], "Hidden layer bias adjustments null");
        assertNotNull(adj.bias[1], "Output layer bias adjustments null");
    }

    @Generated("GitHub Copilot")
    @Test
    public void testDeepNetworkGradientBackpropagation() {
        // 2 Input -> 4 Hidden -> 2 Hidden -> 2 Output (3 Dense Layers)
        Network network = createNetworkWithMultipleHiddenLayers(2, new int[]{4, 2}, 2);
        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5, -0.3};
        double[] expectedOutput = {0.8, 0.2};

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expectedOutput, 1);

        // 3 Dense Layers (2 hidden + 1 output)
        assertEquals(3, adj.weight.length);
        assertEquals(3, adj.bias.length);

        // Prüfe dass alle Layer Gradienten haben
        for (int i = 0; i < 3; i++) {
            assertNotNull(adj.weight[i], "Layer " + i + " weight adjustments null");
            assertNotNull(adj.bias[i], "Layer " + i + " bias adjustments null");
            assertTrue(adj.weight[i].length > 0, "Layer " + i + " hat keine Neuronen");
        }
    }

    @Generated("GitHub Copilot")
    @Test
    public void testGradientMagnitudeReasonable() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] input = {0.5};
        double[] output = {0.5};

        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, output, 1);

        // Gradienten sollten nicht extremer groß sein
        for (double[][] layerWeights : adj.weight) {
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    assertTrue(Math.abs(weight) < 100.0, "Gradient zu groß: " + weight);
                }
            }
        }
    }

    // ===== HELPER METHODS =====

    @Generated("GitHub Copilot")
    private Adjustments computeWithLearningRate(Network network, double lr) {
        GradientDescent gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(lr)
        );

        double[] input = {1.0};
        double[] output = {0.5};

        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, output, 1);
        return adj;
    }

    @Generated("GitHub Copilot")
    private boolean allAdjustmentsZero(Adjustments adj) {
        for (double[][] layerWeights : adj.weight) {
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    if (weight != 0.0) return false;
                }
            }
        }
        for (double[] layerBias : adj.bias) {
            for (double bias : layerBias) {
                if (bias != 0.0) return false;
            }
        }
        return true;
    }

    @Generated("GitHub Copilot")
    private boolean containsNaN(Adjustments adj) {
        for (double[][] layerWeights : adj.weight) {
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    if (Double.isNaN(weight)) return true;
                }
            }
        }
        for (double[] layerBias : adj.bias) {
            for (double bias : layerBias) {
                if (Double.isNaN(bias)) return true;
            }
        }
        return false;
    }

    @Generated("GitHub Copilot")
    private boolean containsInfinity(Adjustments adj) {
        for (double[][] layerWeights : adj.weight) {
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    if (Double.isInfinite(weight)) return true;
                }
            }
        }
        for (double[] layerBias : adj.bias) {
            for (double bias : layerBias) {
                if (Double.isInfinite(bias)) return true;
            }
        }
        return false;
    }

    // ===== NETZWERK-FACTORY METHODEN =====

    @Generated("GitHub Copilot")
    private Network createNetworkWithHiddenLayer(int inputSize, int hiddenSize, int outputSize) {
        // inputSize Input -> hiddenSize Hidden -> outputSize Output
        ActivationFunction relu = ActivationFunction.RELU;
        ActivationFunction linear = ActivationFunction.LINEAR;

        DenseLayer[] layers = new DenseLayer[]{
                new DenseLayer(hiddenSize).withActivationFunction(relu),
                new DenseLayer(outputSize).withActivationFunction(linear)
        };

        return new Network(42, new ConstantWeightInitializer(0.5), inputSize, layers);
    }

    @Generated("GitHub Copilot")
    private Network createNetworkWithMultipleHiddenLayers(@SuppressWarnings("SameParameterValue") int inputSize, int[] hiddenSizes, int outputSize) {
        // inputSize Input -> hiddenSizes[0] Hidden -> hiddenSizes[1] Hidden -> ... -> outputSize Output
        ActivationFunction relu = ActivationFunction.RELU;
        ActivationFunction linear = ActivationFunction.LINEAR;

        // Berechne Anzahl der Dense Layers (alle Hidden Layers + Output Layer)
        DenseLayer[] layers = new DenseLayer[hiddenSizes.length + 1];

        // Hidden Layers mit ReLU
        for (int i = 0; i < hiddenSizes.length; i++) {
            layers[i] = new DenseLayer(hiddenSizes[i]).withActivationFunction(relu);
        }

        // Output Layer mit Linear
        layers[hiddenSizes.length] = new DenseLayer(outputSize).withActivationFunction(linear);

        return new Network(42, new ConstantWeightInitializer(0.5), inputSize, layers);
    }

    // ===== TEST UTILITY CLASSES =====

    @Generated("GitHub Copilot")
    private static class ConstantWeightInitializer implements WeightInitializer {
        private final double weight;

        public ConstantWeightInitializer(double weight) {
            this.weight = weight;
        }

        @Override
        public double nextWeight(java.util.Random random, Network network, int layer) {
            return weight;
        }
    }

    // ===== 5. NUMERISCHES GRADIENT CHECKING =====

    /**
     * Vergleicht den von GradientDescent berechneten analytischen Gradienten für das Gewicht
     * des Output-Neurons mit dem numerisch per Finite-Differences ermittelten Gradienten.
     * Dies beweist die korrekte Berechnung der partiellen Ableitung im einfachsten Fall.
     */
    @Generated("GitHub Copilot")
    @Test
    public void testNumericalGradientCheckOutputWeight() {
        double lr = 0.01;
        double epsilon = 1e-5;

        // 1 Input → 1 Output (LINEAR), w=0.5, b=0 → output=0.5, expected=1.0 → Fehler vorhanden
        Network network = new Network(42, new ConstantWeightInitializer(0.5), 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        DenseNeuron outputNeuron = (DenseNeuron) network.denseLayers[0].neurons[0];
        outputNeuron.bias = 0.0;

        double[] input = {1.0};
        double[] expected = {1.0};

        GradientDescent gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expected, 1);

        // Analytischer Gradient = Adjustment / Lernrate (da batchSize=1)
        double analyticalGradient = adj.weight[0][0][0] / lr;

        // Numerischer Gradient via Central Finite Differences: (f(w+ε) - f(w-ε)) / 2ε
        LossFunction mse = LossFunction.MEAN_SQUARED_ERROR;
        outputNeuron.incoming[0].weight += epsilon;
        double lossPlus = mse.loss(expected, network.feedForward(input).lastOutput());
        outputNeuron.incoming[0].weight -= 2 * epsilon;
        double lossMinus = mse.loss(expected, network.feedForward(input).lastOutput());
        outputNeuron.incoming[0].weight += epsilon; // Gewicht wiederherstellen
        double numericalGradient = (lossPlus - lossMinus) / (2 * epsilon);

        assertEquals(numericalGradient, analyticalGradient, 1e-4,
                "Analytischer Gradient des Output-Gewichts muss dem numerischen entsprechen");
    }

    /**
     * Wie {@link #testNumericalGradientCheckOutputWeight}, aber für den Bias des Output-Neurons.
     */
    @Generated("GitHub Copilot")
    @Test
    public void testNumericalGradientCheckOutputBias() {
        double lr = 0.01;
        double epsilon = 1e-5;

        Network network = new Network(42, new ConstantWeightInitializer(0.5), 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        DenseNeuron outputNeuron = (DenseNeuron) network.denseLayers[0].neurons[0];
        outputNeuron.bias = 0.0;

        double[] input = {1.0};
        double[] expected = {1.0};

        GradientDescent gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expected, 1);
        double analyticalGradient = adj.bias[0][0] / lr;

        LossFunction mse = LossFunction.MEAN_SQUARED_ERROR;
        outputNeuron.bias += epsilon;
        double lossPlus = mse.loss(expected, network.feedForward(input).lastOutput());
        outputNeuron.bias -= 2 * epsilon;
        double lossMinus = mse.loss(expected, network.feedForward(input).lastOutput());
        outputNeuron.bias += epsilon; // Bias wiederherstellen
        double numericalGradient = (lossPlus - lossMinus) / (2 * epsilon);

        assertEquals(numericalGradient, analyticalGradient, 1e-4,
                "Analytischer Gradient des Output-Bias muss dem numerischen entsprechen");
    }

    /**
     * Numerisches Gradient Checking für das Gewicht eines Hidden-Layer-Neurons.
     * Dieser Test verifiziert die Korrektheit der Backpropagation durch mehrere Layer.
     * Netz: 1 Input → 1 Hidden (SIGMOID) → 1 Output (LINEAR).
     */
    @Generated("GitHub Copilot")
    @Test
    public void testNumericalGradientCheckHiddenWeight() {
        double lr = 0.01;
        double epsilon = 1e-5;

        // 1 Input → 1 Hidden (SIGMOID) → 1 Output (LINEAR), alle Gewichte = 0.5, Biases = 0
        Network network = new Network(42, new ConstantWeightInitializer(0.5), 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.SIGMOID),
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        ((DenseNeuron) network.denseLayers[0].neurons[0]).bias = 0.0;
        ((DenseNeuron) network.denseLayers[1].neurons[0]).bias = 0.0;

        double[] input = {1.0};
        double[] expected = {1.0};

        GradientDescent gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expected, 1);

        // Hidden-Layer-Gewicht (Layer 0, Neuron 0, Connection 0) analytisch
        double analyticalGradient = adj.weight[0][0][0] / lr;

        // Numerisch: Perturbation des Hidden-Layer-Gewichts
        LossFunction mse = LossFunction.MEAN_SQUARED_ERROR;
        network.denseLayers[0].neurons[0].incoming[0].weight += epsilon;
        double lossPlus = mse.loss(expected, network.feedForward(input).lastOutput());
        network.denseLayers[0].neurons[0].incoming[0].weight -= 2 * epsilon;
        double lossMinus = mse.loss(expected, network.feedForward(input).lastOutput());
        network.denseLayers[0].neurons[0].incoming[0].weight += epsilon; // Gewicht wiederherstellen
        double numericalGradient = (lossPlus - lossMinus) / (2 * epsilon);

        assertEquals(numericalGradient, analyticalGradient, 1e-4,
                "Backpropagation: Analytischer Gradient des Hidden-Gewichts muss dem numerischen entsprechen");
    }

    // ===== 6. KORREKTHEIT VON adjust() =====

    /**
     * Stellt sicher, dass {@link Adjustments#adjust(Network)} die Gewichte und Biases
     * im Netzwerk tatsächlich verändert (nicht nur berechnet).
     */
    @Generated("GitHub Copilot")
    @Test
    public void testAdjustChangesWeightsAndBias() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        // input={1.0}, expected={1.0} → output≠1.0 (Gewichte 0.5, Bias klein) → Gradients ≠ 0
        double[] input = {1.0};
        double[] expected = {1.0};

        double weightBefore = network.denseLayers[0].neurons[0].incoming[0].weight;
        double biasBefore = ((DenseNeuron) network.denseLayers[0].neurons[0]).bias;

        GradientDescent gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.1));
        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expected, 1);
        adj.adjust(network);

        assertNotEquals(weightBefore, network.denseLayers[0].neurons[0].incoming[0].weight,
                "adjust() muss das Gewicht verändern");
        assertNotEquals(biasBefore, ((DenseNeuron) network.denseLayers[0].neurons[0]).bias,
                "adjust() muss den Bias verändern");
    }

    /**
     * Nach einem einzelnen Gradient-Descent-Schritt (compute + adjust) muss der Loss
     * für denselben Datenpunkt kleiner sein als vorher.
     */
    @Generated("GitHub Copilot")
    @Test
    public void testLossDecreasesAfterOneAdjustStep() {
        // 1 Input → 1 Output (LINEAR), w=0.5, b=0 → output=0.5, expected=1.0
        Network network = new Network(42, new ConstantWeightInitializer(0.5), 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        ((DenseNeuron) network.denseLayers[0].neurons[0]).bias = 0.0;

        double[] input = {1.0};
        double[] expected = {1.0};
        LossFunction mse = LossFunction.MEAN_SQUARED_ERROR;

        double lossBefore = mse.loss(expected, network.feedForward(input).lastOutput());

        GradientDescent gradientDescent = new GradientDescent(mse, new ConstantLearningRate(0.1));
        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expected, 1);
        adj.adjust(network);

        double lossAfter = mse.loss(expected, network.feedForward(input).lastOutput());
        assertTrue(lossAfter < lossBefore,
                "Loss muss nach einem GD-Schritt sinken. Vorher: " + lossBefore + ", Nachher: " + lossAfter);
    }

    // ===== 7. WEITERE VERHALTENSTESTS =====

    /**
     * Wenn die Vorhersage exakt dem erwarteten Wert entspricht (Loss = 0),
     * muss der Gradient überall null sein – das Netz ist bereits optimal.
     */
    @Generated("GitHub Copilot")
    @Test
    public void testPerfectPredictionZeroAdjustments() {
        // w=0.5, b=0, input=2.0 → output = 0.5 * 2.0 = 1.0 = expected → perfekte Vorhersage
        Network network = new Network(42, new ConstantWeightInitializer(0.5), 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        ((DenseNeuron) network.denseLayers[0].neurons[0]).bias = 0.0;

        double[] input = {2.0};
        double[] expected = {1.0}; // 0.5 * 2.0 = 1.0 → perfekt

        GradientDescent gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));
        gradientDescent.epoch(0, 0);
        Adjustments adj = Adjustments.generateEmpty(network);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj, buffer, network, input, expected, 1);

        assertTrue(allAdjustmentsZero(adj),
                "Bei perfekter Vorhersage sollten alle Anpassungen gleich 0 sein");
    }

    /**
     * {@code compute()} darf das Netzwerk nicht verändern. Zwei aufeinanderfolgende Aufrufe
     * mit denselben Eingaben müssen identische Anpassungen liefern.
     */
    @Generated("GitHub Copilot")
    @Test
    public void testComputeIsNonDestructive() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] input = {1.0};
        double[] expected = {0.5};

        GradientDescent gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));

        Adjustments adj1 = Adjustments.generateEmpty(network);
        Adjustments adj2 = Adjustments.generateEmpty(network);
        gradientDescent.epoch(0, 0);
        OutputsDerived buffer = OutputsDerived.generateEmpty(network);
        gradientDescent.compute(adj1, buffer, network, input, expected, 1);
        gradientDescent.epoch(0, 0);
        gradientDescent.compute(adj2, buffer, network, input, expected, 1);

        for (int layer = 0; layer < adj1.weight.length; layer++) {
            assertArrayEquals(adj1.bias[layer], adj2.bias[layer], 1e-15,
                    "compute() darf das Netz nicht verändern – Bias-Adjustments in Layer " + layer + " weichen ab");
            for (int neuron = 0; neuron < adj1.weight[layer].length; neuron++) {
                assertArrayEquals(adj1.weight[layer][neuron],
                        adj2.weight[layer][neuron], 1e-15,
                        "compute() darf das Netz nicht verändern – Gewichts-Adjustments in Layer " + layer
                                + ", Neuron " + neuron + " weichen ab");
            }
        }
    }

    /**
     * Prüft die Vorzeichen der Anpassungen:
     * <ul>
     *   <li>Underprediction (output &lt; expected): Adjustment negativ → Gewicht wächst</li>
     *   <li>Overprediction  (output &gt; expected): Adjustment positiv → Gewicht sinkt</li>
     * </ul>
     * Nur für das Output-Gewicht eines einfachen linearen Netzes ohne Bias geprüft.
     */
    @Generated("GitHub Copilot")
    @Test
    public void testAdjustmentSignMatchesGradientDirection() {
        // w=0.5, b=0, input=2.0, expected=2.0 → output=1.0 < expected → Underprediction
        Network networkUnder = new Network(42, new ConstantWeightInitializer(0.5), 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        ((DenseNeuron) networkUnder.denseLayers[0].neurons[0]).bias = 0.0;

        GradientDescent gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));
        gradientDescent.epoch(0, 0);
        Adjustments adjUnder = Adjustments.generateEmpty(networkUnder);
        OutputsDerived buffer = OutputsDerived.generateEmpty(networkUnder);
        gradientDescent.compute(adjUnder, buffer, networkUnder, new double[]{2.0}, new double[]{2.0}, 1);

        // Adjustment muss negativ sein → weight -= negativeAdj = weight steigt
        assertTrue(adjUnder.weight[0][0][0] < 0,
                "Underprediction: Adjustment des Ausgabe-Gewichts muss negativ sein (→ Gewicht steigt)");

        // w=0.5, b=0, input=2.0, expected=0.5 → output=1.0 > expected → Overprediction
        Network networkOver = new Network(42, new ConstantWeightInitializer(0.5), 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        ((DenseNeuron) networkOver.denseLayers[0].neurons[0]).bias = 0.0;

        gradientDescent.epoch(0, 0);
        Adjustments adjOver = Adjustments.generateEmpty(networkOver);
        gradientDescent.compute(adjOver, buffer, networkOver, new double[]{2.0}, new double[]{0.5}, 1);

        // Adjustment muss positiv sein → weight -= positiveAdj = weight sinkt
        assertTrue(adjOver.weight[0][0][0] > 0,
                "Overprediction: Adjustment des Ausgabe-Gewichts muss positiv sein (→ Gewicht sinkt)");
    }

    /**
     * MAE und MSE berechnen unterschiedliche Gradienten, wenn |error| ≠ 0.5.
     * Bei einem Fehler von 1.0 liefert MSE den doppelten Gradienten gegenüber MAE.
     */
    @Generated("GitHub Copilot")
    @Test
    public void testMAEProducesDifferentGradientThanMSE() {
        // w=0, b=0, input=1.0 → output=0.0, expected=1.0, Fehler=1.0
        // MSE derivedLoss = -2 * 1.0 = -2.0 → Adjustment = -0.02
        // MAE derivedLoss = -1.0          → Adjustment = -0.01
        double lr = 0.01;

        Network mseNet = new Network(42, WeightInitializer.ZERO, 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        ((DenseNeuron) mseNet.denseLayers[0].neurons[0]).bias = 0.0;

        Network maeNet = new Network(42, WeightInitializer.ZERO, 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        ((DenseNeuron) maeNet.denseLayers[0].neurons[0]).bias = 0.0;

        double[] input = {1.0};
        double[] expected = {1.0};

        GradientDescent gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        Adjustments adjMse = Adjustments.generateEmpty(mseNet);
        OutputsDerived buffer = OutputsDerived.generateEmpty(mseNet);
        gradientDescent.compute(adjMse, buffer, mseNet, input, expected, 1);
        double mseAdjustment = adjMse.weight[0][0][0];

        gradientDescent = new GradientDescent(LossFunction.MEAN_ABSOLUTE_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        Adjustments adjMae = Adjustments.generateEmpty(maeNet);
        gradientDescent.compute(adjMae, buffer, maeNet, input, expected, 1);
        double maeAdjustment = adjMae.weight[0][0][0];

        assertNotEquals(mseAdjustment, maeAdjustment,
                "MSE und MAE sollen bei gleichem Fehler ≠ 0.5 unterschiedliche Gradienten liefern");
        // MSE-Gradient soll doppelt so groß sein wie MAE-Gradient
        assertEquals(2.0, mseAdjustment / maeAdjustment, 1e-10,
                "Bei Fehler=1.0 muss MSE-Gradient doppelt so groß sein wie MAE-Gradient");
    }
}