package de.fhdw.knn.data;

public interface Normalizer {

    void normalize(double[][] data);

    void denormalize(double[][] data);

}
