package de.fhdw.knn.config.network;

import de.fhdw.knn.network.layer.DenseLayer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ConfLayerTest {

    @Test
    void testConfLayerCreate() {
        ConfLayer conf = new ConfLayer();
        conf.neurons = 3;
        conf.activationFunction = "LINEAR";

        DenseLayer layer = conf.create();

        assertNotNull(layer);
        assertEquals(3, layer.neurons.length);
    }

    @Test
    void testConfLayerCreateInvalidActivationFunction() {
        ConfLayer conf = new ConfLayer();
        conf.neurons = 3;
        conf.activationFunction = "DOES_NOT_EXIST";

        assertThrows(Exception.class, conf::create);
    }
}