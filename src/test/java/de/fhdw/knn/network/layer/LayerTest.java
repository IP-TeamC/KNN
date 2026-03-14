package de.fhdw.knn.network.layer;

import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.InputNeuron;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

public class LayerTest {

    @Test
    public void testInput() {
        Layer layer = new InputLayer(10);
        assertEquals(10, layer.neurons().length);
        // noinspection CastCanBeRemovedNarrowingVariableType
        assertEquals(((InputLayer) layer).neurons, layer.neurons());
        assertInstanceOf(InputNeuron.class, layer.neurons()[0]);
        assertNotSame(layer.neurons()[0], layer.neurons()[1]);
    }

    @Test
    public void testDense() {
        Layer layer = new DenseLayer(10);
        assertEquals(10, layer.neurons().length);
        // noinspection CastCanBeRemovedNarrowingVariableType
        assertEquals(((DenseLayer) layer).neurons, layer.neurons());
        assertInstanceOf(DenseNeuron.class, layer.neurons()[0]);
        assertNotSame(layer.neurons()[0], layer.neurons()[1]);

        // noinspection CastCanBeRemovedNarrowingVariableType
        DenseLayer denseLayer = (DenseLayer) layer;
        assertNull(denseLayer.neurons[0].incoming);
        assertNull(((DenseNeuron) denseLayer.neurons[0]).activationFunction);
        denseLayer.withActivationFunction(ActivationFunction.SIGMOID);
        assertEquals(ActivationFunction.SIGMOID, ((DenseNeuron) denseLayer.neurons[0]).activationFunction);
    }

    @Test
    public void testCreateDenseLayers() {
        DenseLayer[] denseLayers = DenseLayer.createLayers(ActivationFunction.RELU, ActivationFunction.SOFTPLUS, 10, 10, 2);
        assertEquals(3, denseLayers.length);
        assertEquals(10, denseLayers[0].neurons().length);
        assertEquals(10, denseLayers[1].neurons().length);
        assertEquals(2, denseLayers[2].neurons().length);

        assertEquals(ActivationFunction.RELU, ((DenseNeuron) denseLayers[0].neurons[0]).activationFunction);
        assertEquals(ActivationFunction.RELU, ((DenseNeuron) denseLayers[1].neurons[0]).activationFunction);
        assertEquals(ActivationFunction.SOFTPLUS, ((DenseNeuron) denseLayers[2].neurons[0]).activationFunction);

    }

    @Generated("GitHub Copilot")
    @Test
    public void testCreateLayersSingleLayer() {
        // Ein Eintrag → nur Output Layer mit der outputActivationFunction
        DenseLayer[] layers = DenseLayer.createLayers(ActivationFunction.RELU, ActivationFunction.SIGMOID, 5);
        assertEquals(1, layers.length, "Nur ein Layer erwartet");
        assertEquals(5, layers[0].neurons().length);
        assertEquals(ActivationFunction.SIGMOID, ((DenseNeuron) layers[0].neurons[0]).activationFunction,
                "Einziger Layer muss outputActivationFunction verwenden");
    }

    @Generated("GitHub Copilot")
    @Test
    public void testCreateLayersTwoLayers() {
        // Zwei Einträge → 1 Hidden Layer (hiddenActivation) + 1 Output Layer (outputActivation)
        DenseLayer[] layers = DenseLayer.createLayers(ActivationFunction.RELU, ActivationFunction.SIGMOID, 4, 2);
        assertEquals(2, layers.length);
        assertEquals(ActivationFunction.RELU, ((DenseNeuron) layers[0].neurons[0]).activationFunction,
                "Hidden Layer muss hiddenActivationFunction verwenden");
        assertEquals(ActivationFunction.SIGMOID, ((DenseNeuron) layers[1].neurons[0]).activationFunction,
                "Output Layer muss outputActivationFunction verwenden");
    }

    @Generated("GitHub Copilot")
    @Test
    public void testDenseLayerInitialBias() {
        // Jedes neu erstellte DenseNeuron hat den Bias 0.000_000_000_1
        DenseLayer layer = new DenseLayer(3);
        for (int i = 0; i < 3; i++) {
            DenseNeuron neuron = (DenseNeuron) layer.neurons[i];
            assertEquals(0.0000000001, neuron.bias, 1e-20,
                    "Neuron " + i + " sollte den initialen Bias 1e-10 haben");
        }
    }
}
