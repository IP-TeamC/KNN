package de.fhdw.knn.data;

/**
 * Normalisiert Daten linear zwischen einem Minimum und Maximum
 */
public class MinMaxNormalizer implements Normalizer {

    /**
     * Minimum für den MinMaxNormalizer
     */
    private final double min;

    /**
     * Maximum für den MinMaxNormalizer
     */
    private final double max;

    /**
     * Minima pro Feature
     */
    private double[] dataMin;

    /**
     * Maxima pro Feature
     */
    private double[] dataMax;

    /**
     * Erzeugt aus Min- und Max-Wert einen MinMaxNormalizer.
     *
     * @param min Minimum für den MinMaxNormalizer
     * @param max Maximum für den MinMaxNormalizer
     * @see MinMaxNormalizer
     */
    public MinMaxNormalizer(double min, double max) {
        this.min = min;
        this.max = max;
    }

    /**
     * Normalisiert die Daten.
     *
     * @param data Daten, die normalisiert werden
     */
    @Override
    public void normalize(double[][] data) {
        dataMin = new double[data[0].length];
        dataMax = new double[data[0].length];
        for (int i = 0; i < dataMin.length; i++) {
            dataMin[i] = Integer.MAX_VALUE;
            dataMax[i] = Integer.MIN_VALUE;
        }
        for (double[] ds : data) {
            for (int i = 0; i < ds.length; i++) {
                if (ds[i] < dataMin[i]) {
                    dataMin[i] = ds[i];
                }
                if (ds[i] > dataMax[i]) {
                    dataMax[i] = ds[i];
                }
            }
        }
        double diff = max - min;
        for (double[] ds : data) {
            for (int i = 0; i < ds.length; i++) {
                ds[i] = min + (diff * (ds[i] - dataMin[i]) / (dataMax[i] - dataMin[i]));
            }
        }
    }

    /**
     * Denormalisiert die Daten.
     *
     * @param data Daten, die denormalisiert werden
     */
    @Override
    public void denormalize(double[][] data) {
        double diff = max - min;
        for (double[] ds : data) {
            for (int i = 0; i < ds.length; i++) {
                ds[i] = (ds[i] - min) * (dataMax[i] - dataMin[i]) / diff + dataMin[i];
            }
        }
    }

}
