package de.fhdw.knn.trainer.optimization;

import de.fhdw.knn.TestUtil;
import de.fhdw.knn.network.neuron.DenseNeuron;
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

public class GradientDescentTest {

    private GradientDescent gradientDescent;

    // ===== 1. EINFACHE STRUKTUR-TESTS =====

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
    @Test
    public void testSingleNeuronGradientComputationSuperNeurons() {
        // 1 Input -> 1 Hidden -> 1 Output
        Network network = createNetworkWithHiddenLayer(1, 1, 1);
        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );
        new SuperNeuron(TestUtil.simpleDummyNetwork()).insert(network, 0, 0);
        new SuperNeuron(TestUtil.simpleDummyNetwork()).insert(network, 1, 0);

        double[] input = {1.0};
        double[] expectedOutput = {1.0};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expectedOutput, 1);

        assertNotNull(adj.adjustmentsWeight());
        assertNotNull(adj.adjustmentsBias());
        assertEquals(2, adj.adjustmentsWeight().length);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testSingleNeuronGradientComputationSuperNeuronsSimpler() {
        // 1 Input -> 1 output
        Network network = TestUtil.simpleDummyNetwork();
        gradientDescent = new GradientDescent(
                LossFunction.MEAN_SQUARED_ERROR,
                new ConstantLearningRate(0.01)
        );
        new SuperNeuron(TestUtil.simpleDummyNetwork()).insert(network, 0, 0);

        double[] input = {1.0};
        double[] expectedOutput = {1.0};

        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expectedOutput, 1);

        assertNotNull(adj.adjustmentsWeight());
        assertNotNull(adj.adjustmentsBias());
        assertEquals(1, adj.adjustmentsWeight().length);
    }

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

    @Generated("GitHub Copilot")
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

        gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expected, 1);

        // Analytischer Gradient = Adjustment / Lernrate (da batchSize=1)
        double analyticalGradient = adj.adjustmentsWeight()[0][0][0] / lr;

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

        gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expected, 1);
        double analyticalGradient = adj.adjustmentsBias()[0][0] / lr;

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

        gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expected, 1);

        // Hidden-Layer-Gewicht (Layer 0, Neuron 0, Connection 0) analytisch
        double analyticalGradient = adj.adjustmentsWeight()[0][0][0] / lr;

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

        gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.1));
        gradientDescent.epoch(0, 0);
        gradientDescent.compute(network, input, expected, 1).adjust(network);

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

        gradientDescent = new GradientDescent(mse, new ConstantLearningRate(0.1));
        gradientDescent.epoch(0, 0);
        gradientDescent.compute(network, input, expected, 1).adjust(network);

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

        gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));
        gradientDescent.epoch(0, 0);
        Adjustments adj = gradientDescent.compute(network, input, expected, 1);

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

        gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));

        gradientDescent.epoch(0, 0);
        Adjustments adj1 = gradientDescent.compute(network, input, expected, 1);
        gradientDescent.epoch(0, 0);
        Adjustments adj2 = gradientDescent.compute(network, input, expected, 1);

        for (int layer = 0; layer < adj1.adjustmentsWeight().length; layer++) {
            assertArrayEquals(adj1.adjustmentsBias()[layer], adj2.adjustmentsBias()[layer], 1e-15,
                    "compute() darf das Netz nicht verändern – Bias-Adjustments in Layer " + layer + " weichen ab");
            for (int neuron = 0; neuron < adj1.adjustmentsWeight()[layer].length; neuron++) {
                assertArrayEquals(adj1.adjustmentsWeight()[layer][neuron],
                        adj2.adjustmentsWeight()[layer][neuron], 1e-15,
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

        gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(0.01));
        gradientDescent.epoch(0, 0);
        Adjustments adjUnder = gradientDescent.compute(networkUnder, new double[]{2.0}, new double[]{2.0}, 1);

        // Adjustment muss negativ sein → weight -= negativeAdj = weight steigt
        assertTrue(adjUnder.adjustmentsWeight()[0][0][0] < 0,
                "Underprediction: Adjustment des Ausgabe-Gewichts muss negativ sein (→ Gewicht steigt)");

        // w=0.5, b=0, input=2.0, expected=0.5 → output=1.0 > expected → Overprediction
        Network networkOver = new Network(42, new ConstantWeightInitializer(0.5), 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        ((DenseNeuron) networkOver.denseLayers[0].neurons[0]).bias = 0.0;

        gradientDescent.epoch(0, 0);
        Adjustments adjOver = gradientDescent.compute(networkOver, new double[]{2.0}, new double[]{0.5}, 1);

        // Adjustment muss positiv sein → weight -= positiveAdj = weight sinkt
        assertTrue(adjOver.adjustmentsWeight()[0][0][0] > 0,
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

        gradientDescent = new GradientDescent(LossFunction.MEAN_SQUARED_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        double mseAdjustment = gradientDescent.compute(mseNet, input, expected, 1).adjustmentsWeight()[0][0][0];

        gradientDescent = new GradientDescent(LossFunction.MEAN_ABSOLUTE_ERROR, new ConstantLearningRate(lr));
        gradientDescent.epoch(0, 0);
        double maeAdjustment = gradientDescent.compute(maeNet, input, expected, 1).adjustmentsWeight()[0][0][0];

        assertNotEquals(mseAdjustment, maeAdjustment,
                "MSE und MAE sollen bei gleichem Fehler ≠ 0.5 unterschiedliche Gradienten liefern");
        // MSE-Gradient soll doppelt so groß sein wie MAE-Gradient
        assertEquals(2.0, mseAdjustment / maeAdjustment, 1e-10,
                "Bei Fehler=1.0 muss MSE-Gradient doppelt so groß sein wie MAE-Gradient");
    }
}