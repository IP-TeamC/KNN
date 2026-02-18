package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.loss.LossFunction;

public class GradientDescentTest {

    private GradientDescent gradientDescent;

    // ===== 1. EINFACHE STRUKTUR-TESTS =====

    @Test
    public void testSingleNeuronGradientComputation() {
        // 1 Input -> 2 Hidden -> 1 Output
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {1.0};
        double[] expectedOutput = {1.0};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expectedOutput, 1);

        assertNotNull(adj.adjustmentsWeight());
        assertNotNull(adj.adjustmentsBias());
        assertEquals(2, adj.adjustmentsWeight().length);
    }

    @Test
    public void testMultipleOutputNeurons() {
        // 1 Input -> 3 Hidden -> 3 Output
        Network network = createNetworkWithHiddenLayer(1, 3, 3);
        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5};
        double[] expectedOutput = {0.8, 0.2, 0.5};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expectedOutput, 1);

        assertEquals(2, adj.adjustmentsWeight().length);
        assertEquals(2, adj.adjustmentsBias().length);
        assertEquals(3, adj.adjustmentsWeight()[1].length);
    }

    @Test
    public void testTwoLayerNetworkGradients() {
        // 2 Input -> 3 Hidden -> 1 Output
        Network network = createNetworkWithMultipleHiddenLayers(2, new int[]{3, 2}, 1);
        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5, -0.3};
        double[] expectedOutput = {0.8};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expectedOutput, 1);

        // 3 Layer total: Input -> 3 Hidden -> 2 Hidden -> 1 Output
        assertEquals(3, adj.adjustmentsWeight().length);
        assertEquals(3, adj.adjustmentsBias().length);
    }

    // ===== 2. GRADIENT-EIGENSCHAFTEN =====

    @Test
    public void testGradientProportionalToLearningRate() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);

        Adjustments adj1 = computeWithLearningRate(network, 0.01);
        Adjustments adj2 = computeWithLearningRate(network, 0.02);

        // Bei 2x Lernrate sollten Gradienten ~2x sein
        if (Math.abs(adj1.adjustmentsBias()[1][0]) > 1e-10) {
            double ratio = adj2.adjustmentsBias()[1][0] / adj1.adjustmentsBias()[1][0];
            assertEquals(2.0, ratio, 0.05);
        }
    }

    @Test
    public void testGradientInverselyProportionalToBatchSize() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] input = {1.0};
        double[] output = {0.5};

        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adjBatch1 = gradientDescent.compute(network, input, output, 1);

        gradientDescent.epoch(0, 0);
        Adjustments adjBatch32 = gradientDescent.compute(network, input, output, 32);

        // Mit 32x Batch sollten Adjustments 1/32 sein
        if (Math.abs(adjBatch32.adjustmentsBias()[1][0]) > 1e-10) {
            double ratio = adjBatch1.adjustmentsBias()[1][0] / adjBatch32.adjustmentsBias()[1][0];
            assertEquals(32.0, ratio, 1.0);
        }
    }

    // ===== 3. SPEZIALFÄLLE =====

    @Test
    public void testZeroLearningRateNoChanges() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] input = {1.0};
        double[] output = {0.5};

        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.0)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, output, 1);

        assertTrue(allAdjustmentsZero(adj));
    }

    @Test
    public void testLargeInputsStability() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] largeInput = {1000.0};
        double[] output = {1.0};

        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, largeInput, output, 1);

        assertFalse(containsNaN(adj), "Gradient contains NaN values");
        assertFalse(containsInfinity(adj), "Gradient contains Infinity values");
    }

    @Test
    public void testVerySmallLearningRateStability() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] input = {1.0};
        double[] output = {0.5};

        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(1e-8)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, output, 1);

        assertFalse(containsNaN(adj), "Gradient with tiny learning rate contains NaN");
        assertFalse(containsInfinity(adj), "Gradient with tiny learning rate contains Infinity");
    }

    @Test
    public void testNegativeInputsStability() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] negativeInput = {-100.0};
        double[] output = {0.0};

        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, negativeInput, output, 1);

        assertFalse(containsNaN(adj));
        assertFalse(containsInfinity(adj));
    }

    // ===== 4. BACKPROPAGATION KONSISTENZ =====

    @Test
    public void testMultiLayerGradientFlow() {
        // 2 Input -> 3 Hidden -> 3 Output
        Network network = createNetworkWithHiddenLayer(2, 3, 3);
        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5, -0.3};
        double[] expectedOutput = {0.8, 0.2, 0.5};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expectedOutput, 1);

        // Prüfe dass beide Layer Gradienten haben
        assertEquals(2, adj.adjustmentsWeight().length);
        assertNotNull(adj.adjustmentsWeight()[0], "Hidden layer weight adjustments null");
        assertNotNull(adj.adjustmentsWeight()[1], "Output layer weight adjustments null");
        assertNotNull(adj.adjustmentsBias()[0], "Hidden layer bias adjustments null");
        assertNotNull(adj.adjustmentsBias()[1], "Output layer bias adjustments null");
    }

    @Test
    public void testDeepNetworkGradientBackpropagation() {
        // 2 Input -> 4 Hidden -> 2 Hidden -> 2 Output (3 Dense Layers)
        Network network = createNetworkWithMultipleHiddenLayers(2, new int[]{4, 2}, 2);
        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5, -0.3};
        double[] expectedOutput = {0.8, 0.2};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expectedOutput, 1);

        // 3 Dense Layers (2 hidden + 1 output)
        assertEquals(3, adj.adjustmentsWeight().length);
        assertEquals(3, adj.adjustmentsBias().length);

        // Prüfe dass alle Layer Gradienten haben
        for (int i = 0; i < 3; i++) {
            assertNotNull(adj.adjustmentsWeight()[i], "Layer " + i + " weight adjustments null");
            assertNotNull(adj.adjustmentsBias()[i], "Layer " + i + " bias adjustments null");
            assertTrue(adj.adjustmentsWeight()[i].length > 0, "Layer " + i + " hat keine Neuronen");
        }
    }

    @Test
    public void testGradientMagnitudeReasonable() {
        Network network = createNetworkWithHiddenLayer(1, 2, 1);
        double[] input = {0.5};
        double[] output = {0.5};

        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, output, 1);

        // Gradienten sollten nicht extremer groß sein
        for (double[][] layerWeights : adj.adjustmentsWeight()) {
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    assertTrue(Math.abs(weight) < 100.0, "Gradient zu groß: " + weight);
                }
            }
        }
    }

    // ===== HELPER METHODS =====

    private Adjustments computeWithLearningRate(Network network, double lr) {
        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(lr)
        );

        double[] input = {1.0};
        double[] output = {0.5};

        gradientDescent.epoch(0, 0);
        return gradientDescent.compute(network, input, output, 1);
    }

    private boolean allAdjustmentsZero(Adjustments adj) {
        for (double[][] layerWeights : adj.adjustmentsWeight()) {
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    if (weight != 0.0) return false;
                }
            }
        }
        for (double[] layerBias : adj.adjustmentsBias()) {
            for (double bias : layerBias) {
                if (bias != 0.0) return false;
            }
        }
        return true;
    }

    private boolean containsNaN(Adjustments adj) {
        for (double[][] layerWeights : adj.adjustmentsWeight()) {
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    if (Double.isNaN(weight)) return true;
                }
            }
        }
        for (double[] layerBias : adj.adjustmentsBias()) {
            for (double bias : layerBias) {
                if (Double.isNaN(bias)) return true;
            }
        }
        return false;
    }

    private boolean containsInfinity(Adjustments adj) {
        for (double[][] layerWeights : adj.adjustmentsWeight()) {
            for (double[] neuronWeights : layerWeights) {
                for (double weight : neuronWeights) {
                    if (Double.isInfinite(weight)) return true;
                }
            }
        }
        for (double[] layerBias : adj.adjustmentsBias()) {
            for (double bias : layerBias) {
                if (Double.isInfinite(bias)) return true;
            }
        }
        return false;
    }

    // ===== NETZWERK-FACTORY METHODEN =====

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

    private Network createNetworkWithMultipleHiddenLayers(int inputSize, int[] hiddenSizes, int outputSize) {
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
}