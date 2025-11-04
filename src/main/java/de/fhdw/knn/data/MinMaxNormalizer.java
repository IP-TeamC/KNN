package de.fhdw.knn.data;

public class MinMaxNormalizer implements Normalizer {

    public double min;
    public double max;

    public double dataMin[];
    public double dataMax[];

    public MinMaxNormalizer(double min, double max) {
        this.min = min;
        this.max = max;
    }

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
