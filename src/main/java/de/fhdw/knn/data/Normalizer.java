package de.fhdw.knn.data;

/**
 * Interface zur Implementierung eines Normalizers/Denormalizers
 */
public interface Normalizer {

    /**
     * Normalisiert die Daten.
     *
     * @param data Zu normalisierende Daten. Werden dabei verändert
     */
    void normalize(double[][] data);

    /**
     * Denormalisiert die Daten.
     *
     * @param data Zu denormalisierende Daten. Werden dabei verändert
     */
    void denormalize(double[][] data);

}
