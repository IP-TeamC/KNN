package de.fhdw.knn.trainer.learningrate;

import org.junit.jupiter.api.Test;

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

}
