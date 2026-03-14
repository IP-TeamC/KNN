package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static java.lang.Math.*;
import static org.junit.jupiter.api.Assertions.*;

public class WeightInitializerTest {

    private static final Network network = new Network(
            new InputLayer(10),
            new DenseLayer[]{
                    new DenseLayer(20),
                    new DenseLayer(5)
            });

    @Test
    public void testGlorotUniform() {
        WeightInitializer init = WeightInitializer.GLOROT_UNIFORM;
        Random random = new Random(42);
        for (int i = 0; i < 100; i++) {
            double weight0 = init.nextWeight(random, network, 0);
            assertTrue(weight0 < sqrt(0.2));
            assertTrue(weight0 > -sqrt(0.2));

            double weight1 = init.nextWeight(random, network, 1);
            assertTrue(weight1 < sqrt(0.24));
            assertTrue(weight1 > -sqrt(0.24));
        }
    }

    @Test
    public void testGlorot() {
        WeightInitializer init = WeightInitializer.GLOROT;
        Random random = new Random(42);
        for (int i = 0; i < 100; i++) {
            double weight0 = init.nextWeight(random, network, 0);
            // ca. 99% Konfidenzintervall
            assertTrue(weight0 < 3.29 * sqrt(0.067));
            assertTrue(weight0 > -3.29 * sqrt(0.067));

            double weight1 = init.nextWeight(random, network, 1);
            // 99.9% Konfidenzintervall
            assertTrue(weight1 < 3.29 * sqrt(0.08));
            assertTrue(weight1 > -3.29 * sqrt(0.08));
        }
    }

    @Test
    public void testHeUniform() {
        WeightInitializer init = WeightInitializer.HE_UNIFORM;
        Random random = new Random(42);
        for (int i = 0; i < 100; i++) {
            double weight0 = init.nextWeight(random, network, 0);
            assertTrue(weight0 < sqrt(0.6));
            assertTrue(weight0 > -sqrt(0.6));

            double weight1 = init.nextWeight(random, network, 1);
            assertTrue(weight1 < sqrt(0.3));
            assertTrue(weight1 > -sqrt(0.3));
        }
    }

    @Test
    public void testHe() {
        WeightInitializer init = WeightInitializer.HE;
        Random random = new Random(42);
        for (int i = 0; i < 100; i++) {
            double weight0 = init.nextWeight(random, network, 0);
            // 99.9% Konfidenzintervall
            assertTrue(weight0 < 3.29 * sqrt(0.2));
            assertTrue(weight0 > -3.29 * sqrt(0.2));

            double weight1 = init.nextWeight(random, network, 1);
            // 99.9% Konfidenzintervall
            assertTrue(weight1 < 3.29 * sqrt(0.1));
            assertTrue(weight1 > -3.29 * sqrt(0.1));
        }
    }



}
