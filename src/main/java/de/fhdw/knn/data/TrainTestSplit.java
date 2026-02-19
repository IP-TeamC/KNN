package de.fhdw.knn.data;

/**
 * Aufteilung der Daten in eine Tranings-/Test-DataSet
 */
public class TrainTestSplit {

    /**
     * Trainings-Datensatz
     */
    public DataSet train;
    /**
     * Test-Datensatz
     */
    public DataSet test;

    /**
     * @see TrainTestSplit
     */
    public TrainTestSplit(DataSet train, DataSet test) {
        this.train = train;
        this.test = test;
    }

}
