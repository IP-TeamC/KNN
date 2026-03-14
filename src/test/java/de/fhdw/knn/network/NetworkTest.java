package de.fhdw.knn.network;

import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.connection.Connection;
import de.fhdw.knn.network.connection.WeightInitializer;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.OutputsDerived;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

public class NetworkTest {

    // Testet nur Code, der nicht bereits zuvor in anderen Tests des Packages getestet wurde

    @Test
    public void testFeedforward() {
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                DenseLayer.createLayers(ActivationFunction.LINEAR, ActivationFunction.TANH, 1, 1));
        network.denseLayers[0].neurons[0].incoming[0].weight = 42;
        network.denseLayers[0].neurons[0].incoming[1].weight = 24;
        network.denseLayers[1].neurons[0].incoming[0].weight = 0.25;

        double[] prediction = network.feedForward(new double[]{-3, 5.4}).lastOutput();
        assertEquals(0.71629787, prediction[0], 1e-8);
    }

    @Test
    public void testNonOverridingConnector() {
        InputLayer inputLayer = new InputLayer(3);
        DenseLayer denseLayer1 = new DenseLayer(2).withActivationFunction(ActivationFunction.RELU);
        DenseLayer denseLayer2 = new DenseLayer(5).withActivationFunction(ActivationFunction.TANH);
        denseLayer1.neurons[0].incoming = new Connection[]{
                new Connection(3),
                new Connection(1),
                new Connection(5)
        };
        denseLayer2.neurons[3].incoming = new Connection[]{
                new Connection(4),
                new Connection(2),
        };
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM,
                inputLayer, denseLayer1, denseLayer2);
        network.predictSingles(new double[][]{
                new double[]{1, 2, 3},
                new double[]{4, 2, 0}
        });

        assertEquals(3, network.denseLayers[0].neurons[0].incoming[0].weight);
        assertEquals(1, network.denseLayers[0].neurons[0].incoming[1].weight);
        assertEquals(5, network.denseLayers[0].neurons[0].incoming[2].weight);
        assertEquals(4, network.denseLayers[1].neurons[3].incoming[0].weight);
        assertEquals(2, network.denseLayers[1].neurons[3].incoming[1].weight);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testFeedForwardSimpleComputation() {
        // 2 Eingaben → 1 Ausgabe, Linear, manuell gesetzte Gewichte
        Network network = new Network(42, WeightInitializer.ZERO, 2,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        DenseNeuron outputNeuron = (DenseNeuron) network.denseLayers[0].neurons[0];
        outputNeuron.incoming[0].weight = 2.0;
        outputNeuron.incoming[1].weight = 3.0;
        outputNeuron.bias = 0.0;

        OutputsDerived result = network.feedForward(new double[]{1.0, 1.0});
        // 2*1 + 3*1 = 5
        assertEquals(5.0, result.lastOutput()[0], 1e-10);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testFeedForwardDeepNetwork() {
        // 1 Eingabe → 1 Hidden (Linear) → 1 Ausgabe (Linear), manuell gesetzte Gewichte
        Network network = new Network(42, WeightInitializer.ZERO, 1,
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR),
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        DenseNeuron hiddenNeuron = (DenseNeuron) network.denseLayers[0].neurons[0];
        DenseNeuron outputNeuron = (DenseNeuron) network.denseLayers[1].neurons[0];
        hiddenNeuron.incoming[0].weight = 1.0;
        hiddenNeuron.bias = 0.0;
        outputNeuron.incoming[0].weight = 2.0;
        outputNeuron.bias = 0.0;

        OutputsDerived result = network.feedForward(new double[]{3.0});
        assertEquals(3.0, result.output()[0][0], 1e-10); // Hidden Layer: 1*3 = 3
        assertEquals(6.0, result.lastOutput()[0], 1e-10); // Output Layer: 2*3 = 6
    }

    @Generated("GitHub Copilot")
    @Test
    public void testFeedForwardAndPredictSinglesConsistency() {
        // feedForward() und predictSingles() müssen identische Ergebnisse liefern
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 3,
                new DenseLayer(2).withActivationFunction(ActivationFunction.SIGMOID),
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        double[] input = {1.0, -0.5, 2.0};
        double feedForwardOutput = network.feedForward(input).lastOutput()[0];
        double predictedSingle = network.predictSingles(new double[][]{input})[0];
        assertEquals(feedForwardOutput, predictedSingle, 1e-10);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testDeterministicWeightInitialization() {
        // Gleiches Seed → identische Gewichte bei gleicher Netzstruktur
        Network n1 = new Network(123, WeightInitializer.GLOROT_UNIFORM, 3,
                new DenseLayer(4).withActivationFunction(ActivationFunction.RELU),
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));
        Network n2 = new Network(123, WeightInitializer.GLOROT_UNIFORM, 3,
                new DenseLayer(4).withActivationFunction(ActivationFunction.RELU),
                new DenseLayer(1).withActivationFunction(ActivationFunction.LINEAR));

        for (int layer = 0; layer < n1.denseLayers.length; layer++) {
            for (int neuron = 0; neuron < n1.denseLayers[layer].neurons.length; neuron++) {
                Connection[] c1 = n1.denseLayers[layer].neurons[neuron].incoming;
                Connection[] c2 = n2.denseLayers[layer].neurons[neuron].incoming;
                for (int conn = 0; conn < c1.length; conn++) {
                    assertEquals(c1[conn].weight, c2[conn].weight, 1e-15,
                            "Gewichte differieren bei Layer=" + layer + ", Neuron=" + neuron + ", Verbindung=" + conn);
                }
            }
        }
    }

    @Generated("GitHub Copilot")
    @Test
    public void testPredictOutputShape() {
        // predict() muss für n Eingaben n Ausgabe-Arrays mit jeweils k Werten zurückgeben
        Network network = new Network(42, WeightInitializer.GLOROT_UNIFORM, 2,
                new DenseLayer(3).withActivationFunction(ActivationFunction.RELU),
                new DenseLayer(2).withActivationFunction(ActivationFunction.SIGMOID));
        double[][] inputs = {{1.0, 0.0}, {0.5, -0.5}, {2.0, 1.0}};
        double[][] outputs = network.predict(inputs);

        assertEquals(3, outputs.length, "predict() soll für jede Eingabe eine Ausgabe liefern");
        for (double[] output : outputs) {
            assertEquals(2, output.length, "Jede Ausgabe muss 2 Werte (Output-Neuronen) enthalten");
        }
    }
}
