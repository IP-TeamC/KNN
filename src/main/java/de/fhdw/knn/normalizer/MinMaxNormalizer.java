package de.fhdw.knn.normalizer;

public class MinMaxNormalizer {

    public double min;
    public double max;

    public double dataMin = Integer.MAX_VALUE;
    public double dataMax = Integer.MIN_VALUE;

    public MinMaxNormalizer(double min, double max) {
        this.min = min;
        this.max = max;
    }

    public void normalize(double[][] data) {
        for (double[] ds : data) {
            for (double d : ds) {
                if (d < dataMin) {
                    dataMin = d;
                }
                if (d > dataMax) {
                    dataMax = d;
                }
            }
        }
        double diff = max - min;
        double dataDiff = dataMax - dataMin;
        for (double[] ds : data) {
            for (int i = 0; i < ds.length; i++) {
                ds[i] = min + (diff * (ds[i] - dataMin) / dataDiff);
            }
        }
    }

    public void denormalize(double[][] data) {
        double diff = max - min;
        double dataDiff = dataMax - dataMin;
        for (double[] ds : data) {
            for (int i = 0; i < ds.length; i++) {
                ds[i] = (ds[i] - min) * dataDiff / diff + dataMin;
            }
        }
    }

}
