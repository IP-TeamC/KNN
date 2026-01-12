package de.fhdw.knn.util;

import org.jfree.chart.renderer.PaintScale;
import java.awt.Color;

public class SmoothWeightPaintScale implements PaintScale {
    private final double lowerBound;
    private final double upperBound;
    private static final double GUARD = 0.2;

    private static final Color[] POSITIVE_GRADIENT = {
            new Color(255, 69, 0),
            new Color(255, 165, 0),
            new Color(255, 255, 0),
            new Color(255, 255, 255)
    };

    private static final Color[] NEGATIVE_GRADIENT = {
            new Color(44, 238, 14),
            new Color(0, 220, 220),
            new Color(0, 0, 209),
            new Color(0, 0, 0)
    };

    public SmoothWeightPaintScale(double lowerBound, double upperBound) {
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    @Override
    public double getLowerBound() {
        return lowerBound;
    }

    @Override
    public double getUpperBound() {
        return upperBound;
    }

    @Override
    public java.awt.Paint getPaint(double value) {

        if (Math.abs(value) < GUARD) {
            return Color.GRAY;
        }

        double absValue = Math.abs(value);
        double maxAbs = Math.max(Math.abs(lowerBound), Math.abs(upperBound));

        double normalized = (absValue / maxAbs) * 3.0;
        normalized = Math.min(normalized, 3.0);

        int segment = Math.min((int) normalized, 2);
        double t = normalized - segment;

        Color[] gradient = value > 0 ? POSITIVE_GRADIENT : NEGATIVE_GRADIENT;

        return interpolateColor(gradient[segment], gradient[segment + 1], t);
    }

    private Color interpolateColor(Color c1, Color c2, double t) {
        t = Math.max(0.0, Math.min(1.0, t));

        int r = (int) (c1.getRed() + t * (c2.getRed() - c1.getRed()));
        int g = (int) (c1.getGreen() + t * (c2.getGreen() - c1.getGreen()));
        int b = (int) (c1.getBlue() + t * (c2.getBlue() - c1.getBlue()));

        return new Color(r, g, b);
    }
}