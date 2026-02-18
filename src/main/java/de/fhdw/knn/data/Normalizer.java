package de.fhdw.knn.data;

/**
 * Interface zur Implementierung eines Normalizers/Denormalizers
 */
public interface Normalizer {

    /**
     * Normalisiert die Daten.
     *
     * @param data kann dabei verändert werden
     */
    void normalize(double[][] data);

    /**
     * Denormalisiert die Daten.
     *
     * @param data kann dabei verändert werden
     */
    void denormalize(double[][] data);

}
