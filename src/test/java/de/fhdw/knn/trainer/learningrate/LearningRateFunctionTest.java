package de.fhdw.knn.trainer.learningrate;

import org.junit.jupiter.api.Test;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

public class LearningRateFunctionTest {

    @Test
    public void testConstant() {
        LearningRateFunction constant = new ConstantLearningRate(0.25);
        for (int i = 0; i < 100; i++) {
            assertEquals(0.25, constant.calc(i, Math.random()));
        }
    }

    @Test
    public void testDecay() {
        LearningRateFunction decay = new DecayLearningRate(1, 0.9);
        double expected = 1;
        for (int i = 0; i < 100; i++) {
            assertEquals(expected, decay.calc(i, Math.random()));
            expected *= 0.9;
        }
    }

    @Test
    public void testSoftstart() {
        LearningRateFunction softstart = new SoftstartLearningRate(1, 0.5, 3);
        assertEquals(0.5, softstart.calc(1, Math.random()));
        assertEquals(0.5, softstart.calc(2, Math.random()));
        assertEquals(0.5, softstart.calc(3, Math.random()));
        for (int i = 4; i < 100; i++) {
            assertEquals(1, softstart.calc(i, Math.random()));
        }
    }

    @Test
    public void testSoftstartDecay() {
        LearningRateFunction softstartDecay = new SoftstartDecayLearningRate(1, 0.5, 3, 0.9);
        assertEquals(0.5, softstartDecay.calc(1, Math.random()));
        assertEquals(0.5, softstartDecay.calc(2, Math.random()));
        assertEquals(0.5, softstartDecay.calc(3, Math.random()));
        double expected = 1;
        for (int i = 4; i < 100; i++) {
            assertEquals(expected, softstartDecay.calc(i, Math.random()));
            expected *= 0.9;
        }
    }

    @Generated("GitHub Copilot")
    @Test
    public void testDecayActualValues() {
        // Decay halbiert die LR in jedem Schritt – exakte Werte werden geprüft
        LearningRateFunction decay = new DecayLearningRate(1.0, 0.5);
        assertEquals(1.0,   decay.calc(0, 0), 1e-10);
        assertEquals(0.5,   decay.calc(1, 0), 1e-10);
        assertEquals(0.25,  decay.calc(2, 0), 1e-10);
        assertEquals(0.125, decay.calc(3, 0), 1e-10);
    }

    @Generated("GitHub Copilot")
    @Test
    public void testDecayFactorOneKeepsRateConstant() {
        // decay=1.0 → LR bleibt jede Epoche konstant
        LearningRateFunction decay = new DecayLearningRate(0.5, 1.0);
        for (int i = 0; i < 10; i++) {
            assertEquals(0.5, decay.calc(i, 0), 1e-10,
                    "LR sollte bei decay=1.0 konstant bleiben, Epoche " + i);
        }
    }

    @Generated("GitHub Copilot")
    @Test
    public void testSoftstartBoundaryEpoch() {
        // Epoche <= softEpochs → gedämpfte LR; Epoche > softEpochs → volle LR
        LearningRateFunction softstart = new SoftstartLearningRate(1.0, 0.1, 5);
        assertEquals(0.1, softstart.calc(5, 0), 1e-10, "Grenzepoche (= softEpochs) sollte noch gedämpft sein");
        assertEquals(1.0, softstart.calc(6, 0), 1e-10, "Erste normale Epoche (> softEpochs) sollte volle LR haben");
    }
}
