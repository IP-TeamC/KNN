package de.fhdw.knn.data;

/**
 * Paar aus 2 Werten verschiedener Typen
 *
 * @param <X> Beliebiger Datentyp
 * @param <Y> Beliebiger Datentpy
 */
public class Pair<X, Y> {

    /**
     * 1. Wert des Paars (X)
     */
    public X x;
    /**
     * 2. Wert des Paars (Y)
     */
    public Y y;

    /**
     * Erzeugt ein Pair aus zwei beliebigen Datentypen.
     *
     * @param x Beliebige Werte
     * @param y Beliebige Werte
     * @see Pair
     */
    public Pair(X x, Y y) {
        this.x = x;
        this.y = y;
    }

}
