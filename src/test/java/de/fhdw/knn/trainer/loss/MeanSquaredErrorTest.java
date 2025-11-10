package de.fhdw.knn.trainer.loss;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Mean Squared Error Loss Function Tests")
public class MeanSquaredErrorTest {

    private static final double EPSILON = 1e-5;
    private static final double TOLERANCE = 1e-4;
    private final MeanSquaredError mse = new MeanSquaredError();

    // ===== GRUNDLEGENDE EIGENSCHAFTEN =====

    @Test
    @DisplayName("Perfekte Vorhersage sollte Loss = 0 ergeben")
    public void testPerfectPredictionZeroLoss() {
        double[] expected = {1.0, 0.5, 0.2};
        double[] predicted = {1.0, 0.5, 0.2};

        double loss = mse.loss(expected, predicted);
        assertEquals(0.0, loss, 1e-10);
    }

    @Test
    @DisplayName("Loss sollte nicht-negativ sein")
    public void testLossNonNegative() {
        double[] expected = {1.0, 0.0, -1.0};
        double[] predicted = {0.5, -0.5, 0.5};

        double loss = mse.loss(expected, predicted);
        assertTrue(loss >= 0, "Loss sollte nicht-negativ sein, ist aber: " + loss);
    }

    @Test
    @DisplayName("Loss ist symmetrisch in Vorhersage-Richtung")
    public void testSymmetricError() {
        double[] expected = {1.0};

        double loss1 = mse.loss(expected, new double[]{0.0}); // error = 1.0
        double loss2 = mse.loss(expected, new double[]{2.0}); // error = -1.0

        assertEquals(loss1, loss2, 1e-10);
    }

    @Test
    @DisplayName("Loss skaliert quadratisch mit Error")
    public void testQuadraticScaling() {
        double[] expected = {1.0};

        double loss1 = mse.loss(expected, new double[]{0.0}); // error = 1.0, loss = 1.0
        double loss2 = mse.loss(expected, new double[]{-1.0}); // error = 2.0, loss = 4.0

        assertEquals(4.0, loss2 / loss1, 0.01);
    }

    @Test
    @DisplayName("Loss mittelt über alle Outputs")
    public void testAveragingOverOutputs() {
        double[] expected = {1.0, 1.0};
        double[] predicted = {0.0, 0.0};
        // Error für beide: 1.0, sum = 1.0 + 1.0 = 2.0, average = 1.0

        double loss = mse.loss(expected, predicted);
        assertEquals(1.0, loss, 1e-10);
    }

    // ===== NUMERISCHE EIGENSCHAFTEN =====

    @Test
    @DisplayName("Numerisches Gradient Checking")
    public void testGradientNumerical() {
        double[] expected = {0.8, 0.2};
        double[] predicted = {0.7, 0.3};

        double analyticalGradient = mse.derivedLoss(expected, predicted);
        double numericalGradient = computeNumericalGradient(
                p -> mse.loss(expected, p),
                predicted
        );

        assertEquals(numericalGradient, analyticalGradient, TOLERANCE,
                "Gradient Check fehlgeschlagen: analytical=" + analyticalGradient +
                        " vs numerical=" + numericalGradient);
    }

    @Test
    @DisplayName("Gradient sollte in Richtung von Error zeigen")
    public void testGradientDirection() {
        double[] expected = {1.0};
        double[] predicted = {0.5};

        // expected > predicted, error = 0.5
        // Gradient sollte negativ sein (predicted zu niedrig)
        double gradient = mse.derivedLoss(expected, predicted);
        assertTrue(gradient < 0, "Gradient sollte negativ sein für unterprognostizierte Werte");
    }

    @Test
    @DisplayName("Sehr großer Error produces großen Gradient")
    public void testLargeErrorLargeGradient() {
        double[] expected = {10.0};

        double grad1 = Math.abs(mse.derivedLoss(expected, new double[]{9.0}));
        double grad2 = Math.abs(mse.derivedLoss(expected, new double[]{5.0}));

        assertTrue(grad2 > grad1, "Größerer Error sollte größeren Gradient haben");
    }

    // ===== MEHRERE OUTPUTS =====

    @Test
    @DisplayName("Gradient für mehrere Outputs")
    public void testGradientMultipleOutputs() {
        double[] expected = {0.5, 0.5, 0.5};
        double[] predicted = {0.4, 0.5, 0.6};

        double gradient = mse.derivedLoss(expected, predicted);

        assertFalse(Double.isNaN(gradient), "Gradient sollte nicht NaN sein");
        assertFalse(Double.isInfinite(gradient), "Gradient sollte nicht Infinity sein");
    }

    @Test
    @DisplayName("Loss mit einzelnem Output")
    public void testSingleOutput() {
        double[] expected = {0.7};
        double[] predicted = {0.3};

        double loss = mse.loss(expected, predicted);
        // (0.7 - 0.3)^2 = 0.16
        assertEquals(0.16, loss, 1e-10);
    }

    @Test
    @DisplayName("Loss mit vielen Outputs")
    public void testManyOutputs() {
        int size = 100;
        double[] expected = new double[size];
        double[] predicted = new double[size];

        for (int i = 0; i < size; i++) {
            expected[i] = 1.0;
            predicted[i] = 0.5;
        }

        double loss = mse.loss(expected, predicted);
        // Jeder Output: (1.0 - 0.5)^2 = 0.25
        // Average: 0.25
        assertEquals(0.25, loss, 1e-10);
    }

    // ===== EDGE CASES =====

    @Test
    @DisplayName("Sehr kleine Errors")
    public void testVerySmallErrors() {
        double[] expected = {0.0};
        double[] predicted = {1e-10};

        double loss = mse.loss(expected, predicted);
        assertTrue(loss > 0 && loss < 1e-15);
    }

    @Test
    @DisplayName("Sehr große Values")
    public void testVeryLargeValues() {
        double[] expected = {1e6};
        double[] predicted = {1e6 - 1};

        double loss = mse.loss(expected, predicted);
        assertFalse(Double.isNaN(loss));
        assertFalse(Double.isInfinite(loss));
    }

    @Test
    @DisplayName("Negative Values")
    public void testNegativeValues() {
        double[] expected = {-1.0, -0.5};
        double[] predicted = {-0.5, -1.0};

        double loss = mse.loss(expected, predicted);
        assertEquals(0.5, loss, 1e-10);
    }

    @Test
    @DisplayName("Gemischte positive und negative Values")
    public void testMixedSigns() {
        double[] expected = {1.0, -1.0};
        double[] predicted = {0.0, 0.0};

        double loss = mse.loss(expected, predicted);
        assertEquals(1.0, loss, 1e-10);
    }

    @Test
    @DisplayName("Null-Like Arrays mit Zeros")
    public void testZeroArrays() {
        double[] expected = {0.0, 0.0};
        double[] predicted = {0.0, 0.0};

        double loss = mse.loss(expected, predicted);
        assertEquals(0.0, loss, 1e-10);
    }

    // ===== GRADIENT KONSISTENZ =====

    @Test
    @DisplayName("Gradient bei Zero-Error")
    public void testGradientZeroError() {
        double[] expected = {1.0};
        double[] predicted = {1.0};

        double gradient = mse.derivedLoss(expected, predicted);
        assertEquals(0.0, gradient, 1e-10);
    }

    @Test
    @DisplayName("Gradient Richtung: Overprediction")
    public void testGradientOverprediction() {
        double[] expected = {0.0};
        double[] predicted = {1.0};

        // expected < predicted, error = -1.0
        // -2 * (-1.0) = 2.0 → positiv
        double gradient = mse.derivedLoss(expected, predicted);
        assertTrue(gradient > 0);
    }

    @Test
    @DisplayName("Gradient Richtung: Underprediction")
    public void testGradientUnderprediction() {
        double[] expected = {1.0};
        double[] predicted = {0.0};

        // expected > predicted, error = 1.0
        // -2 * (1.0) = -2.0 → negativ
        double gradient = mse.derivedLoss(expected, predicted);
        assertTrue(gradient < 0);
    }

    @Test
    @DisplayName("Gradient Magnitude proportional zu Error")
    public void testGradientMagnitudeProportional() {
        double[] expected = {1.0};
        double[] pred1 = {0.9};
        double[] pred2 = {0.8};

        double grad1 = Math.abs(mse.derivedLoss(expected, pred1));
        double grad2 = Math.abs(mse.derivedLoss(expected, pred2));

        // error1 = 0.1, grad1 = -2 * 0.1 = -0.2
        // error2 = 0.2, grad2 = -2 * 0.2 = -0.4
        assertEquals(grad1 * 2, grad2, 1e-10);
    }

    // ===== STABILITÄT =====

    @Test
    @DisplayName("Keine NaN bei normalen Werten")
    public void testNoNaN() {
        double[] expected = {0.5, 0.5, 0.5};
        double[] predicted = {0.4, 0.5, 0.6};

        double loss = mse.loss(expected, predicted);
        double gradient = mse.derivedLoss(expected, predicted);

        assertFalse(Double.isNaN(loss));
        assertFalse(Double.isNaN(gradient));
    }

    @Test
    @DisplayName("Keine Infinity bei normalen Werten")
    public void testNoInfinity() {
        double[] expected = {1e10};
        double[] predicted = {1e10 + 1};

        double loss = mse.loss(expected, predicted);
        double gradient = mse.derivedLoss(expected, predicted);

        assertFalse(Double.isInfinite(loss));
        assertFalse(Double.isInfinite(gradient));
    }

    // ===== HELPER METHODS =====

    /**
     * Numerisches Gradient Checking mit Finite Differences
     */
    private double computeNumericalGradient(java.util.function.Function<double[], Double> lossFunction, double[] point) {
        double sum = 0;

        for (int i = 0; i < point.length; i++) {
            double[] pointPlus = point.clone();
            double[] pointMinus = point.clone();

            pointPlus[i] += EPSILON;
            pointMinus[i] -= EPSILON;

            double lossPlus = lossFunction.apply(pointPlus);
            double lossMinus = lossFunction.apply(pointMinus);

            double gradient = (lossPlus - lossMinus) / (2.0 * EPSILON);
            sum += gradient;
        }

        return sum / point.length;
    }
}