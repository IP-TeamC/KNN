package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.network.activation.LinearActivationFunction;
import de.fhdw.knn.network.activation.ReLUActivationFunction;
import de.fhdw.knn.trainer.learningrate.ConstantLearningRate;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.trainer.learningrate.LearningRateFunction;
import de.fhdw.knn.trainer.loss.LossFunction;

public class GradientDescentTest {

    private GradientDescent gradientDescent;

    // ===== 1. EINFACHE STRUKTUR-TESTS =====

    @Test
    public void testSingleNeuronGradientComputation() {
        Network network = createSingleNeuronNetwork();
        gradientDescent = new GradientDescent(
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {1.0};
        double[] expectedOutput = {1.0};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(input, expectedOutput, 1);

        assertNotNull(adj.adjustmentsWeight());
        assertNotNull(adj.adjustmentsBias());
        assertEquals(1, adj.adjustmentsWeight().length);
    }

    @Test
    public void testTwoLayerNetworkGradients() {
        Network network = createTwoLayerNetwork(2, 3, 1);
        gradientDescent = new GradientDescent(
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5, -0.3};
        double[] expectedOutput = {0.8};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(input, expectedOutput, 1);

        assertEquals(2, adj.adjustmentsWeight().length);
        assertEquals(2, adj.adjustmentsBias().length);
    }

    // ===== 2. GRADIENT-EIGENSCHAFTEN =====

    @Test
    public void testGradientProportionalToLearningRate() {
        Network network = createSingleNeuronNetwork();

        Adjustments adj1 = computeWithLearningRate(network, 0.01);
        Adjustments adj2 = computeWithLearningRate(network, 0.02);

        // Bei 2x Lernrate sollten Gradienten ~2x sein
        if (Math.abs(adj1.adjustmentsBias()[0][0]) > 1e-10) {
            double ratio = adj2.adjustmentsBias()[0][0] / adj1.adjustmentsBias()[0][0];
            assertEquals(2.0, ratio, 0.05);
        }
    }

    @Test
    public void testGradientInverselyProportionalToBatchSize() {
        Network network = createSingleNeuronNetwork();
        double[] input = {1.0};
        double[] output = {0.5};

        gradientDescent = new GradientDescent(
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adjBatch1 = gradientDescent.compute(input, output, 1);

        gradientDescent.epoch(0, 0);
        Adjustments adjBatch32 = gradientDescent.compute(input, output, 32);

        // Mit 32x Batch sollten Adjustments 1/32 sein
        if (Math.abs(adjBatch32.adjustmentsBias()[0][0]) > 1e-10) {
            double ratio = adjBatch1.adjustmentsBias()[0][0] / adjBatch32.adjustmentsBias()[0][0];
            assertEquals(32.0, ratio, 1.0);
        }
    }

    // ===== 3. SPEZIALFÄLLE =====

    @Test
    public void testZeroLearningRateNoChanges() {
        Network network = createSingleNeuronNetwork();
        double[] input = {1.0};
        double[] output = {0.5};

        gradientDescent = new GradientDescent(
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.0)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(input, output, 1);

        assertTrue(allAdjustmentsZero(adj));
    }

    @Test
    public void testLargeInputsStability() {
        Network network = createSingleNeuronNetwork();
        double[] largeInput = {1000.0};
        double[] output = {1.0};

        gradientDescent = new GradientDescent(
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(largeInput, output, 1);

        assertFalse(containsNaN(adj), "Gradient contains NaN values");
        assertFalse(containsInfinity(adj), "Gradient contains Infinity values");
    }

    @Test
    public void testVerySmallLearningRateStability() {
        Network network = createSingleNeuronNetwork();
        double[] input = {1.0};
        double[] output = {0.5};

        gradientDescent = new GradientDescent(
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(1e-8)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(input, output, 1);

        assertFalse(containsNaN(adj), "Gradient with tiny learning rate contains NaN");
        assertFalse(containsInfinity(adj), "Gradient with tiny learning rate contains Infinity");
    }

    @Test
    public void testNegativeInputsStability() {
        Network network = createSingleNeuronNetwork();
        double[] negativeInput = {-100.0};
        double[] output = {0.0};

        gradientDescent = new GradientDescent(
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(negativeInput, output, 1);

        assertFalse(containsNaN(adj));
        assertFalse(containsInfinity(adj));
    }

    // ===== 4. BACKPROPAGATION KONSISTENZ =====

    @Test
    public void testMultiLayerGradientFlow() {
        // 3-Layer Netzwerk: 2 Input -> 3 Hidden -> 2 Output
        Network network = createTwoLayerNetwork(2, 3, 2);
        gradientDescent = new GradientDescent(
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        double[] input = {0.5, -0.3};
        double[] expectedOutput = {0.8, 0.2};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(input, expectedOutput, 1);

        // Prüfe dass alle Layer Gradienten haben
        for (int layer = 0; layer < adj.adjustmentsWeight().length; layer++) {
            assertNotNull(adj.adjustmentsWeight()[layer], "Layer " + layer + " weight adjustments null");
            assertNotNull(adj.adjustmentsBias()[layer], "Layer " + layer + " bias adjustments null");

            // Prüfe Dimensionen
            assertTrue(adj.adjustmentsWeight()[layer].length > 0, "Layer " + layer + " hat keine Neuronen");
        }
    }

    @Test
    public void testGradientMagnitudeReasonable() {
        Network network = createSingleNeuronNetwork();
        double[] input = {0.5};
        double[] output = {0.5};

        gradientDescent = new GradientDescent(
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(input, output, 1);

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
                network,
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(lr)
        );

        double[] input = {1.0};
        double[] output = {0.5};

        gradientDescent.epoch(0, 0);
        return gradientDescent.compute(input, output, 1);
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

    private Network createSingleNeuronNetwork() {
        // 1 Input -> 1 Output
        ActivationFunction linear = new LinearActivationFunction();
        DenseLayer[] layers = new DenseLayer[]{
                new DenseLayer(1).withActivationFunction(linear)
        };

        return new Network(42, new ConstantWeightInitializer(0.5), 1, layers);
    }

    private Network createTwoLayerNetwork(int inputSize, int hiddenSize, int outputSize) {
        // inputSize Input -> hiddenSize Hidden -> outputSize Output
        ActivationFunction relu = new ReLUActivationFunction();
        ActivationFunction linear = new LinearActivationFunction();

        DenseLayer[] layers = new DenseLayer[]{
                new DenseLayer(hiddenSize).withActivationFunction(relu),
                new DenseLayer(outputSize).withActivationFunction(linear)
        };

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