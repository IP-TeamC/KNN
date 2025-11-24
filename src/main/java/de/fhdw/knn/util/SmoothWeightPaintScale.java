package de.fhdw.knn.util;

import org.jfree.chart.renderer.PaintScale;
import java.awt.Color;

public class SmoothWeightPaintScale implements PaintScale {

    private final double lowerBound;
    private final double upperBound;

    public SmoothWeightPaintScale(double lowerBound, double upperBound) {
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    @Override
    public double getLowerBound() { return lowerBound; }

    @Override
    public double getUpperBound() { return upperBound; }

    @Override
    public Color getPaint(double value) {
        if (value == 0.0) {
            return Color.DARK_GRAY; // 0 bleibt dunkelgrau
        }

        double absValue = Math.abs(value);
        double maxAbs = Math.max(Math.abs(lowerBound), Math.abs(upperBound));

        // Definiere drei gleiche Segmente
        double segment = maxAbs / 3.0;

        if (absValue <= segment) {
            // Erste 1/3 des Bereichs → Grün
            double t = absValue / segment; // 0..1 innerhalb Segment
            return interpolateColor(Color.GREEN, Color.ORANGE, t);
        } else if (absValue <= 2 * segment) {
            // Mittleres Segment → Orange
            double t = (absValue - segment) / segment; // 0..1 innerhalb Segment
            return interpolateColor(Color.ORANGE, Color.ORANGE, t); // optional nur Orange
        } else {
            // Letztes Segment → Orange → Rot
            double t = (absValue - 2 * segment) / segment; // 0..1 innerhalb Segment
            return interpolateColor(Color.ORANGE, Color.RED, t);
        }
    }

    private Color interpolateColor(Color c1, Color c2, double t) {
        int r = (int)(c1.getRed() + t * (c2.getRed() - c1.getRed()));
        int g = (int)(c1.getGreen() + t * (c2.getGreen() - c1.getGreen()));
        int b = (int)(c1.getBlue() + t * (c2.getBlue() - c1.getBlue()));
        return new Color(r, g, b);
    }
}