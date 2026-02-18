package de.fhdw.knn.data;

/**
 * Paar aus 2 Werten verschiedener Typen
 */
public class Pair<X, Y> {

    public X x;
    public Y y;

    /**
     * @see Pair
     */
    public Pair(X x, Y y) {
        this.x = x;
        this.y = y;
    }

}
