package de.fhdw.knn.trainer.loss;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MeanAbsoluteErrorTest {

    private final LossFunction mae = LossFunction.MEAN_ABSOLUTE_ERROR;

    @Test
    public void testPerfect() {
        double[] expected = new double[]{15.4, 20};
        double[] predicted = new double[]{15.4, 20};
        double loss = mae.loss(expected, predicted);
        assertEquals(0.0, loss, 1e-10);
    }

    @Test
    public void testLinear() {
        double[] expected1 = new double[]{15.4, 20};
        double[] predicted1 = new double[]{16.4, 21};
        double loss1 = mae.loss(expected1, predicted1);

        double[] expected2 = new double[]{15.4, 20};
        double[] predicted2 = new double[]{17.4, 22};
        double loss2 = mae.loss(expected2, predicted2);

        assertEquals(1.0, loss1, 1e-10);
        assertEquals(2.0, loss2, 1e-10);
    }

    @Test
    public void testNonNegative() {
        double[] expected = new double[]{16.4, 21};
        double[] predicted = new double[]{15.4, 20};
        double loss = mae.loss(expected, predicted);
        assertEquals(1.0, loss, 1e-10);
    }

    @Test
    public void testAverage() {
        double[] expected = new double[]{16.4, 22};
        double[] predicted = new double[]{15.4, 20};
        double loss = mae.loss(expected, predicted);
        assertEquals(1.5, loss, 1e-10);
    }

    @Test
    public void testGradient() {
        double[] expected = new double[]{15.4};
        double[] predicted = new double[]{16.4};
        double loss = mae.derivedLoss(expected, predicted, 0);
        assertEquals(1, loss, 1e-10);
    }

    @Test
    public void testGradientNegative() {
        double[] expected = new double[]{15.4};
        double[] predicted = new double[]{14.4};
        double loss = mae.derivedLoss(expected, predicted, 0);
        assertEquals(-1, loss, 1e-10);
    }

    @Test
    public void testTotal() {
        double[] expected1 = new double[]{15.4, 20};
        double[] predicted1 = new double[]{16.4, 22};
        double[] expected2 = new double[]{15.4, 20};
        double[] predicted2 = new double[]{19.4, 23};

        double totalLoss = mae.totalLoss(new double[][]{expected1, expected2}, new double[][]{predicted1, predicted2});
        assertEquals(2.5, totalLoss, 1e-10);
    }

}
