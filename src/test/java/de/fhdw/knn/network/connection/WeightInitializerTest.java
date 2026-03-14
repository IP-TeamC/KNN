package de.fhdw.knn.network.connection;

import de.fhdw.knn.network.Network;
import de.fhdw.knn.network.layer.DenseLayer;
import de.fhdw.knn.network.layer.InputLayer;
import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;
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

    @Generated("GitHub Copilot")
    @Test
    public void testZeroInitializer() {
        // ZERO initialisiert alle Gewichte mit dem Wert 0
        WeightInitializer zero = WeightInitializer.ZERO;
        Random random = new Random(42);
        for (int i = 0; i < 50; i++) {
            assertEquals(0.0, zero.nextWeight(random, network, 0), 1e-15,
                    "ZERO-Initializer sollte 0.0 liefern (Layer 0, Iteration " + i + ")");
            assertEquals(0.0, zero.nextWeight(random, network, 1), 1e-15,
                    "ZERO-Initializer sollte 0.0 liefern (Layer 1, Iteration " + i + ")");
        }
    }

    @Generated("GitHub Copilot")
    @Test
    public void testDeterministicInitialization() {
        // Gleicher Zufallsgenerator-Seed → identische Gewichtssequenz
        WeightInitializer init = WeightInitializer.GLOROT_UNIFORM;
        Random random1 = new Random(99);
        Random random2 = new Random(99);
        for (int i = 0; i < 50; i++) {
            assertEquals(init.nextWeight(random1, network, 0),
                         init.nextWeight(random2, network, 0), 1e-15,
                    "Gleicher Seed muss gleiche Gewichte liefern, Iteration " + i);
        }
    }

    @Generated("GitHub Copilot")
    @Test
    public void testGlorotUniformProducesVariedWeights() {
        // GLOROT_UNIFORM soll unterschiedliche Gewichte erzeugen (kein konstanter Wert)
        WeightInitializer init = WeightInitializer.GLOROT_UNIFORM;
        Random random = new Random(42);
        double first = init.nextWeight(random, network, 0);
        boolean foundDifferent = false;
        for (int i = 0; i < 30; i++) {
            if (abs(init.nextWeight(random, network, 0) - first) > 1e-10) {
                foundDifferent = true;
                break;
            }
        }
        assertTrue(foundDifferent, "GLOROT_UNIFORM sollte verschiedene Gewichte erzeugen");
    }
}
