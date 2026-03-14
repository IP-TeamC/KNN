package de.fhdw.knn.trainer.loss;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import javax.annotation.processing.Generated;

import static org.junit.jupiter.api.Assertions.*;

@DisplayName("Binary Cross Entropy Loss Function Tests")
public class BinaryCrossEntropyLossTest {

    private static final double EPSILON = 1e-5;
    private static final double TOLERANCE = 1e-3;
    private final LossFunction bce = LossFunction.CROSS_ENTROPY_LOSS;

    @Test
    public void testPerfect() {
        double[] expected = new double[]{0, 1};
        double[] predicted = new double[]{0, 1};
        double loss = bce.loss(expected, predicted);
        assertEquals(0.0, loss, 1e-10);
    }

    @Test
    public void testDoubleDiff() {
        double[] expected1 = new double[]{0.154, 0.20};
        double[] predicted1 = new double[]{0.164, 0.21};
        double loss1 = bce.loss(expected1, predicted1);

        double[] expected2 = new double[]{0.154, 0.20};
        double[] predicted2 = new double[]{0.174, 0.22};
        double loss2 = bce.loss(expected2, predicted2);

        assertEquals(0.465331729, loss1, 1e-6);
        assertEquals(0.466308109, loss2, 1e-6);
    }

    @Test
    public void testBad() {
        double[] expected = new double[]{0, 1};
        double[] predicted = new double[]{0.9, 0.05};
        double loss = bce.loss(expected, predicted);
        assertEquals(2.64915868, loss, 1e-6);
    }

    @Test
    public void testGradientPositive() {
        // mit steigender Vorhersage wird der Verlust größer
        double[] expected = new double[]{0};
        double[] predicted = new double[]{0.7};
        double loss = bce.derivedLoss(expected, predicted, 0);
        assertEquals(3.33333333, loss, 1e-6);
    }

    @Test
    public void testGradientNegative() {
        // mit steigender Vorhersage, wird der Verlust geringer
        double[] expected = new double[]{1};
        double[] predicted = new double[]{0.05};
        double loss = bce.derivedLoss(expected, predicted, 0);
        assertEquals(-20, loss, 1e-10);
    }

    @Test
    public void testTotal() {
        double[] expected1 = new double[]{0, 1};
        double[] predicted1 = new double[]{0.9, 0.05};
        double[] expected2 = new double[]{1, 0};
        double[] predicted2 = new double[]{0.05, 0.7};

        double totalLoss = bce.totalLoss(new double[][]{expected1, expected2}, new double[][]{predicted1, predicted2});
        assertEquals(2.37450561, totalLoss, 1e-6);
    }

    // ===== GRUNDLEGENDE EIGENSCHAFTEN =====

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Perfekte Vorhersage sollte sehr kleinen Loss haben")
    public void testPerfectPredictionVerySmallLoss() {
        double[] expected = {1.0, 0.0};
        double[] predicted = {0.9999, 0.0001};

        double loss = bce.loss(expected, predicted);
        assertTrue(loss < 0.01, "Loss für gute Vorhersage sollte klein sein: " + loss);
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Loss sollte nicht-negativ sein")
    public void testLossNonNegative() {
        double[] expected = {1.0, 0.0, 1.0};
        double[] predicted = {0.9, 0.1, 0.8};

        double loss = bce.loss(expected, predicted);
        assertTrue(loss >= 0, "BCE Loss sollte nicht-negativ sein, ist aber: " + loss);
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Falsche Vorhersage hat höheren Loss als gute")
    public void testWrongPredictionHighLoss() {
        double[] expected = {1.0};
        double lossGood = bce.loss(expected, new double[]{0.99});
        double lossBad = bce.loss(expected, new double[]{0.01});

        assertTrue(lossBad > lossGood,
                "Falsche Vorhersage sollte höheren Loss haben: good=" + lossGood + " bad=" + lossBad);
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Loss ist symmetrisch bei 0.5 Prediction")
    public void testSymmetryAt0Point5() {
        double loss1 = bce.loss(new double[]{1.0}, new double[]{0.5});
        double loss2 = bce.loss(new double[]{0.0}, new double[]{0.5});

        assertEquals(loss1, loss2, TOLERANCE, "Loss sollte symmetrisch bei 0.5 sein");
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Mittelt über alle Outputs")
    public void testAveragingOverOutputs() {
        double[] expected1 = {1.0};
        double[] predicted1 = {0.9};
        double loss1 = bce.loss(expected1, predicted1);

        double[] expected2 = {1.0, 1.0};
        double[] predicted2 = {0.9, 0.9};
        double loss2 = bce.loss(expected2, predicted2);

        assertEquals(loss1, loss2, TOLERANCE, "Gleiche Vorhersagen sollten gleichen Loss ergeben");
    }

    // ===== NUMERISCHE EIGENSCHAFTEN =====

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Numerisches Gradient Checking")
    public void testGradientNumerical() {
        double[] expected = {0.8, 0.2};
        double[] predicted = {0.7, 0.3};

        double analyticalGradient = bce.derivedLoss(expected, predicted, 0) + bce.derivedLoss(expected, predicted, 1);
        double numericalGradient = computeNumericalGradient(
                p -> bce.loss(expected, p),
                predicted
        );

        assertEquals(numericalGradient, analyticalGradient, TOLERANCE,
                "Gradient Check fehlgeschlagen: analytical=" + analyticalGradient +
                        " vs numerical=" + numericalGradient);
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Gradient Richtung: Expected=1, Prediction zu niedrig")
    public void testGradientDirectionLowPrediction() {
        double[] expected = {1.0};
        double[] predicted = {0.1};

        double gradient = bce.derivedLoss(expected, predicted, 0);
        assertTrue(gradient < 0, "Gradient sollte negativ sein (prediction zu niedrig)");
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Gradient Richtung: Expected=0, Prediction zu hoch")
    public void testGradientDirectionHighPrediction() {
        double[] expected = {0.0};
        double[] predicted = {0.9};

        double gradient = bce.derivedLoss(expected, predicted, 0);
        assertTrue(gradient > 0, "Gradient sollte positiv sein (prediction zu hoch)");
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Größere Fehler sollten größere Gradienten haben")
    public void testLargerErrorLargerGradient() {
        double[] expected = {1.0};
        double grad1 = Math.abs(bce.derivedLoss(expected, new double[]{0.9}, 0));
        double grad2 = Math.abs(bce.derivedLoss(expected, new double[]{0.1}, 0));

        assertTrue(grad2 > grad1, "Größerer Error sollte größeren Gradient haben");
    }

    // ===== MEHRERE OUTPUTS =====

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Gradient für mehrere Outputs")
    public void testGradientMultipleOutputs() {
        double[] expected = {1.0, 0.0, 1.0};
        double[] predicted = {0.8, 0.2, 0.9};

        double gradient = bce.derivedLoss(expected, predicted, 2);

        assertFalse(Double.isNaN(gradient), "Gradient sollte nicht NaN sein");
        assertFalse(Double.isInfinite(gradient), "Gradient sollte nicht Infinity sein");
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Loss mit einzelnem Output - expected=1")
    public void testSingleOutputExpected1() {
        double[] expected = {1.0};
        double[] predicted = {0.7};

        double loss = bce.loss(expected, predicted);
        // -1.0 * log(0.7 + DELTA) ≈ 0.357
        assertTrue(loss > 0.3 && loss < 0.4);
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Loss mit einzelnem Output - expected=0")
    public void testSingleOutputExpected0() {
        double[] expected = {0.0};
        double[] predicted = {0.3};

        double loss = bce.loss(expected, predicted);
        // -1.0 * log(1 - 0.3 + DELTA) ≈ 0.357
        assertTrue(loss > 0.3 && loss < 0.4);
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Loss mit vielen Outputs")
    public void testManyOutputs() {
        int size = 100;
        double[] expected = new double[size];
        double[] predicted = new double[size];

        for (int i = 0; i < size; i++) {
            expected[i] = 1.0;
            predicted[i] = 0.9;
        }

        double loss = bce.loss(expected, predicted);
        assertFalse(Double.isNaN(loss));
        assertFalse(Double.isInfinite(loss));
        assertTrue(loss > 0);
    }

    // ===== NUMERICAL STABILITY (DELTA PROTECTION) =====

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("DELTA verhindert Log(0) bei predicted=1")
    public void testDeltaProtectionPredicted1() {
        double[] expected = {1.0};
        double[] predicted = {1.0};

        double loss = bce.loss(expected, predicted);
        assertFalse(Double.isNaN(loss), "DELTA sollte Log(0) verhindern");
        assertFalse(Double.isInfinite(loss), "Loss sollte nicht Infinity sein");
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("DELTA verhindert Log(0) bei predicted=0")
    public void testDeltaProtectionPredicted0() {
        double[] expected = {0.0};
        double[] predicted = {0.0};

        double loss = bce.loss(expected, predicted);
        assertFalse(Double.isNaN(loss), "DELTA sollte Log(0) verhindern");
        assertFalse(Double.isInfinite(loss), "Loss sollte nicht Infinity sein");
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Numerische Stabilität bei extremen Werten")
    public void testStabilityExtremes() {
        double[] expected = {1.0, 0.0};
        double[] predicted = {0.9999999, 0.0000001};

        double loss = bce.loss(expected, predicted);
        double gradient = bce.derivedLoss(expected, predicted, 1);

        assertFalse(Double.isNaN(loss), "Loss sollte nicht NaN sein");
        assertFalse(Double.isInfinite(loss), "Loss sollte nicht Infinity sein");
        assertFalse(Double.isNaN(gradient), "Gradient sollte nicht NaN sein");
        assertFalse(Double.isInfinite(gradient), "Gradient sollte nicht Infinity sein");
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Numerische Stabilität bei sehr nahe an 0 und 1")
    public void testStabilityNearBoundaries() {
        double[] expected = {1.0, 0.0, 1.0, 0.0};
        double[] predicted = {1.0 - 1e-10, 1e-10, 0.9999, 0.0001};

        double loss = bce.loss(expected, predicted);
        assertFalse(Double.isNaN(loss));
        assertFalse(Double.isInfinite(loss));
    }

    // ===== EDGE CASES =====

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Alle Expected=1")
    public void testAllExpected1() {
        double[] expected = {1.0, 1.0, 1.0};
        double[] predicted = {0.8, 0.9, 0.7};

        double loss = bce.loss(expected, predicted);
        assertTrue(loss > 0);
        assertFalse(Double.isNaN(loss));
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Alle Expected=0")
    public void testAllExpected0() {
        double[] expected = {0.0, 0.0, 0.0};
        double[] predicted = {0.2, 0.1, 0.3};

        double loss = bce.loss(expected, predicted);
        assertTrue(loss > 0);
        assertFalse(Double.isNaN(loss));
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Gemischte Expected Values")
    public void testMixedExpected() {
        double[] expected = {1.0, 0.0, 1.0, 0.0, 1.0};
        double[] predicted = {0.9, 0.1, 0.8, 0.2, 0.7};

        double loss = bce.loss(expected, predicted);
        assertTrue(loss > 0);
        assertFalse(Double.isNaN(loss));
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Prediction=0.5 (neutral)")
    public void testNeutralPrediction() {
        double[] expected = {1.0};
        double[] predicted = {0.5};

        double loss = bce.loss(expected, predicted);
        assertTrue(loss > 0);
        assertFalse(Double.isNaN(loss));
    }

    // ===== GRADIENT KONSISTENZ =====

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Gradient bei Perfect Prediction")
    public void testGradientPerfectPrediction() {
        double[] expected = {1.0};
        double[] predicted = {0.9999};

        double gradient = bce.derivedLoss(expected, predicted, 0);
        // Sollte sehr klein sein
        assertTrue(Math.abs(gradient) < 1.001, "Gradient sollte klein bei guter Prediction sein");
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Gradient Magnitude steigt mit falscher Prediction")
    public void testGradientMagnitudeIncreases() {
        double[] expected = {1.0};
        double grad1 = Math.abs(bce.derivedLoss(expected, new double[]{0.9}, 0));
        double grad2 = Math.abs(bce.derivedLoss(expected, new double[]{0.5}, 0));
        double grad3 = Math.abs(bce.derivedLoss(expected, new double[]{0.1}, 0));

        assertTrue(grad1 < grad2 && grad2 < grad3,
                "Gradient magnitude sollte mit Error steigen: " + grad1 + " < " + grad2 + " < " + grad3);
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Expected=0 und Expected=1 Gradienten haben unterschiedliche Richtung")
    public void testGradientOppositeDirections() {
        double[] predicted = {0.3};

        double grad1 = bce.derivedLoss(new double[]{1.0}, predicted, 0);
        double grad2 = bce.derivedLoss(new double[]{0.0}, predicted, 0);

        assertTrue((grad1 > 0) != (grad2 > 0), "Gradienten sollten unterschiedliche Richtung haben");
    }

    // ===== STABILITÄT =====

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Keine NaN bei normalen Werten")
    public void testNoNaN() {
        double[] expected = {0.5, 0.5, 1.0, 0.0};
        double[] predicted = {0.4, 0.6, 0.9, 0.1};

        double loss = bce.loss(expected, predicted);
        double gradient = bce.derivedLoss(expected, predicted, 3);

        assertFalse(Double.isNaN(loss));
        assertFalse(Double.isNaN(gradient));
    }

    @Generated("GitHub Copilot")
    @Test
    @DisplayName("Keine Infinity bei normalen Werten")
    public void testNoInfinity() {
        double[] expected = {1.0, 0.0, 1.0};
        double[] predicted = {0.99, 0.01, 0.95};

        double loss = bce.loss(expected, predicted);
        double gradient = bce.derivedLoss(expected, predicted, 2);

        assertFalse(Double.isInfinite(loss));
        assertFalse(Double.isInfinite(gradient));
    }

    // ===== HELPER METHODS =====

    /**
     * Numerisches Gradient Checking mit Finite Differences
     */
    @Generated("GitHub Copilot")
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