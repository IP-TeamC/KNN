package de.fhdw.knn.network.layer;

import de.fhdw.knn.network.activation.ActivationFunction;
import de.fhdw.knn.network.neuron.DenseNeuron;
import de.fhdw.knn.network.neuron.InputNeuron;
import org.junit.jupiter.api.Test;

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

}
